import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent


def _load(fname, modname):
    path = ROOT / fname
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, modname):
    try:
        return _load(fname, modname)
    except Exception as exc:
        print(
            f"INFO optional_provider {fname} unavailable {type(exc).__name__}")
        return None


baseline1 = _load("15_ConvTranspose3d_BatchNorm_Subtract.py",
                  "k_baseline1_l2_15")
baseline2 = _load_optional("base_15_ConvTranspose3d_BatchNorm_Subtract.py",
                           "k_baseline2_l2_15")
optimized = _load("opt_15_ConvTranspose3d_BatchNorm_Subtract.py",
                  "k_optimized_l2_15")


class TorchRef(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        return x - x.mean(dim=(2, 3, 4), keepdim=True)


_BENCH_SHAPES = [
    ("small_direct", 2, 4, 8, 4, 8, 8, 3, 2, 1),
    ("medium_acl", 4, 8, 16, 8, 16, 16, 3, 2, 1),
    ("target", 16, 16, 32, 16, 32, 32, 3, 2, 1),
]
_SHAPES = {s[0]: s for s in _BENCH_SHAPES}
_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_MODEL_CACHE = {}


def _sync():
    torch.npu.synchronize()


def _make_input(shape, dtype=torch.float32):
    label, B, Cin, Cout, D, H, W, K, stride, padding = shape
    torch.manual_seed(1234)
    return torch.rand((B, Cin, D, H, W), device="npu", dtype=dtype)


def _make_model(provider, shape, dtype=torch.float32):
    label, B, Cin, Cout, D, H, W, K, stride, padding = shape
    key = (provider, Cin, Cout, K, stride, padding, str(dtype))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if provider == "torch":
        cls = TorchRef
    elif provider == "baseline1":
        cls = baseline1.ModelNew
    elif provider == "baseline2":
        if baseline2 is None:
            return None
        cls = baseline2.ModelNew
    else:
        cls = optimized.ModelNew
    torch.manual_seed(0)
    model = cls(Cin, Cout, K, stride, padding).to(device="npu",
                                                  dtype=dtype).eval()
    _MODEL_CACHE[key] = model
    return model


def _run_provider(provider, shape, x=None):
    model = _make_model(provider, shape,
                        x.dtype if x is not None else torch.float32)
    if model is None:
        return None
    if x is None:
        x = _make_input(shape)
    with torch.no_grad():
        return model(x)


def _max_diff(a, b):
    return float((a - b).abs().max().detach().cpu())


def _check_provider(provider, shape, ref=None, x=None):
    label = shape[0]
    if provider == "baseline2" and baseline2 is None:
        print(f"TEST {provider} {label} SKIP optional_unavailable")
        return True
    try:
        if x is None:
            x = _make_input(shape)
        if ref is None:
            ref = _run_provider("torch", shape, x)
        out = _run_provider(provider, shape, x)
        _sync()
        if out is None:
            print(f"TEST {provider} {label} SKIP optional_unavailable")
            return True
        diff = _max_diff(out, ref)
        ok = diff <= 2e-3
        if ok:
            print(f"TEST {provider} {label} PASS max_diff={diff:.6g}")
            return True
        if provider == "baseline2":
            print(
                f"TEST {provider} {label} SKIP value_mismatch max_diff={diff:.6g}"
            )
            return True
        print(f"TEST {provider} {label} MISMATCH max_diff={diff:.6g}")
        return False
    except Exception as exc:
        if provider == "baseline2":
            print(f"TEST {provider} {label} SKIP runtime_{type(exc).__name__}")
            return True
        print(
            f"TEST {provider} {label} FAIL runtime_{type(exc).__name__}: {exc}"
        )
        return False


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        x = _make_input(shape)
        ref = _run_provider("torch", shape, x)
        _sync()
        for provider in ["baseline1", "baseline2", "optimized"]:
            passed = _check_provider(provider, shape, ref=ref, x=x)
            ok = ok and (passed if provider == "optimized" else True)
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_one(provider, label):
    shape = _SHAPES[label]
    if provider == "baseline2" and baseline2 is None:
        return float("inf")
    x = _make_input(shape)
    try:
        # Correctness is tested separately; this catches provider runtime failures cleanly.
        def fn():
            _run_provider(provider, shape, x)

        for _ in range(5):
            fn()
        _sync()
        try:
            return triton.testing.do_bench(fn,
                                           warmup=10,
                                           rep=30,
                                           return_mode="mean")
        except Exception:
            import time
            times = []
            for _ in range(30):
                t0 = time.perf_counter()
                fn()
                _sync()
                times.append((time.perf_counter() - t0) * 1000.0)
            return sum(times) / len(times)
    except Exception as exc:
        print(
            f"INFO bench {provider} {label} inf runtime_{type(exc).__name__}")
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
        plot_name="convtranspose3d_batchnorm_subtract",
        args={},
    ))
def bench(label, provider):
    return _bench_one(provider, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not args.test and not args.bench:
        args.test = args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        bench.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
