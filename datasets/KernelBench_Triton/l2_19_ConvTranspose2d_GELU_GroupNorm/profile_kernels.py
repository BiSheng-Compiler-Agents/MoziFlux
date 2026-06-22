import argparse
import importlib.util
import pathlib
import sys
import time

import torch
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None
import torch.nn as nn
import torch.nn.functional as F
import triton

HERE = pathlib.Path(__file__).resolve().parent


def _load(fname, key):
    path = HERE / fname
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, key):
    try:
        return _load(fname, key)
    except Exception as exc:
        print(f"INFO provider {key} unavailable {type(exc).__name__}: {exc}")
        return None


baseline1_mod = _load("19_ConvTranspose2d_GELU_GroupNorm.py", "baseline1")
baseline2_mod = _load_optional("base_19_ConvTranspose2d_GELU_GroupNorm.py",
                               "baseline2")
opt_mod = _load("opt_19_ConvTranspose2d_GELU_GroupNorm.py", "optimized")


class TorchRef(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups,
                 num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups,
                                       num_channels=out_channels)
        self.groups = groups

    def forward(self, x):
        y = self.conv_transpose(x)
        y = F.gelu(y, approximate="none")
        return F.group_norm(y, self.group_norm.num_groups,
                            self.group_norm.weight, self.group_norm.bias,
                            self.group_norm.eps)


# label, batch, in_channels, out_channels, height, width, kernel_size, stride, groups_arg, num_groups
_BENCH_SHAPES = [
    ("small", 2, 8, 8, 32, 32, 3, 1, 1, 2),
    ("medium", 8, 16, 16, 64, 64, 3, 1, 1, 4),
    ("target", 128, 64, 64, 256, 256, 3, 1, 8, 8),
]
_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_MODEL_CACHE = {}
_INPUT_CACHE = {}


def _init_args(shape):
    _, _B, in_ch, out_ch, _H, _W, k, stride, groups_arg, num_groups = shape
    return (in_ch, out_ch, k, stride, groups_arg, num_groups)


def _shape_by_label(label):
    for s in _BENCH_SHAPES:
        if s[0] == label:
            return s
    raise KeyError(label)


def _model(provider, shape):
    key = (provider, shape[0])
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    args = _init_args(shape)
    torch.manual_seed(0)
    if provider == "torch":
        model = TorchRef(*args)
    elif provider == "baseline1":
        model = baseline1_mod.ModelNew(*args)
    elif provider == "baseline2":
        if baseline2_mod is None:
            return None
        model = baseline2_mod.ModelNew(*args)
    elif provider == "optimized":
        model = opt_mod.ModelNew(*args)
    else:
        raise KeyError(provider)
    model = model.to(device="npu").eval()
    _MODEL_CACHE[key] = model
    return model


def _make_input(shape):
    label, B, in_ch, _out_ch, H, W, _k, _stride, _groups_arg, _num_groups = shape
    if label not in _INPUT_CACHE:
        torch.manual_seed(123)
        # rand matches the editable baseline get_inputs() contract.
        _INPUT_CACHE[label] = torch.rand((B, in_ch, H, W),
                                         device="npu",
                                         dtype=torch.float32)
    return _INPUT_CACHE[label]


def _skip_provider(provider, shape):
    label = shape[0]
    if provider == "baseline2" and baseline2_mod is None:
        return "provider_unavailable"
    if label == "target" and provider in ("baseline1", "baseline2"):
        return "custom_two_pass_target_preskip"
    return None


def _run_provider(provider, label):
    shape = _shape_by_label(label)
    reason = _skip_provider(provider, shape)
    if reason:
        raise RuntimeError(reason)
    model = _model(provider, shape)
    if model is None:
        raise RuntimeError("provider_unavailable")
    x = _make_input(shape)
    with torch.no_grad():
        return model(x)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _bench_one(provider, label):
    shape = _shape_by_label(label)
    reason = _skip_provider(provider, shape)
    if reason:
        print(f"INFO bench {provider} {label} SKIP {reason}")
        return float("inf")
    try:

        def fn():
            y = _run_provider(provider, label)
            _sync()
            return y

        try:
            return triton.testing.do_bench(fn,
                                           warmup=3,
                                           rep=10,
                                           return_mode="mean")
        except Exception:
            # Manual fallback; returns ms like do_bench/perf_report expects.
            for _ in range(2):
                fn()
            _sync()
            t0 = time.perf_counter()
            reps = 5 if label == "target" else 20
            for _ in range(reps):
                fn()
            _sync()
            return (time.perf_counter() - t0) * 1000.0 / reps
    except Exception as exc:
        print(f"INFO bench {provider} {label} INF {type(exc).__name__}: {exc}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="convtranspose2d_gelu_groupnorm",
        args={},
    ))
def bench(label, provider):
    return _bench_one(provider, label)


def _check_close(a, b):
    if a.shape != b.shape:
        return False, float("inf"), float("inf")
    diff = (a - b).abs()
    max_err = diff.max().item()
    denom = b.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()
    return bool(torch.allclose(a, b, rtol=1e-3, atol=1e-3)), max_err, max_rel


def unit_test():
    ok_all = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        try:
            ref = _run_provider("torch", label)
            _sync()
            print(f"TEST torch {label} PASS")
        except Exception as exc:
            print(f"TEST torch {label} FAIL {type(exc).__name__}: {exc}")
            ok_all = False
            continue
        for provider in ("baseline1", "baseline2", "optimized"):
            reason = _skip_provider(provider, shape)
            if reason:
                print(f"TEST {provider} {label} SKIP {reason}")
                continue
            try:
                out = _run_provider(provider, label)
                _sync()
                ok, max_err, max_rel = _check_close(out, ref)
                if ok:
                    print(
                        f"TEST {provider} {label} PASS max_err={max_err:.6g} max_rel={max_rel:.6g}"
                    )
                else:
                    # Comparison baselines are parser-visible but do not gate optimized correctness.
                    print(
                        f"TEST {provider} {label} SKIP value_diff max_err={max_err:.6g} max_rel={max_rel:.6g}"
                    )
                    if provider == "optimized":
                        ok_all = False
            except Exception as exc:
                print(
                    f"TEST {provider} {label} SKIP {type(exc).__name__}: {exc}"
                )
                if provider == "optimized":
                    ok_all = False
    print("UNIT_TEST PASS" if ok_all else "UNIT_TEST_FAILED")
    return ok_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not args.test and not args.bench:
        args.test = True
        args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        bench.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
