import argparse
import importlib.util
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

ROOT = pathlib.Path(__file__).resolve().parent


def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, name):
    try:
        return _load(fname, name)
    except Exception as exc:
        print(f"INFO optional_provider {fname} unavailable {type(exc).__name__}")
        return None


_input_mod = _load("37_Matmul_Swish_Sum_GroupNorm.py", "k_input_37_Matmul_Swish_Sum_GroupNorm")
_base_mod = _load_optional("base_37_Matmul_Swish_Sum_GroupNorm.py", "k_base_37_Matmul_Swish_Sum_GroupNorm")
_opt_mod = _load("opt_37_Matmul_Swish_Sum_GroupNorm.py", "k_opt_37_Matmul_Swish_Sum_GroupNorm")


class TorchRef(nn.Module):
    def __init__(self, in_features=512, out_features=1024, num_groups=32, bias_shape=None):
        super().__init__()
        if bias_shape is None:
            bias_shape = (out_features,)
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        z = self.matmul(x)
        z = F.silu(z) + self.bias
        return F.group_norm(z, self.group_norm.num_groups, self.group_norm.weight, self.group_norm.bias, self.group_norm.eps)


# label, batch, in_features, out_features, groups.  Covers direct and persistent optimized dispatch.
_BENCH_SHAPES = [
    ("small_direct", 32, 512, 1024, 32),
    ("medium_direct", 512, 1024, 2048, 32),
    ("irregular_direct", 129, 768, 1536, 48),
    ("largeC_direct", 32, 1024, 4096, 64),
    ("persistent_large", 8192, 1024, 4096, 64),
    ("target", 32768, 1024, 4096, 64),
]
_SHAPES = {label: vals for (label, *vals) in _BENCH_SHAPES}
_MODEL_CACHE = {}


def _seed():
    torch.manual_seed(0)
    if hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed_all(0)
        except Exception:
            pass


def _init_args(in_features, out_features, groups):
    return (in_features, out_features, groups, (out_features,))


def _model(provider, init_args):
    key = (provider, tuple(init_args))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    _seed()
    if provider == "torch":
        model = TorchRef(*init_args)
    elif provider == "baseline1":
        model = _input_mod.ModelNew(*init_args)
    elif provider == "baseline2":
        if _base_mod is None:
            return None
        model = _base_mod.ModelNew(*init_args)
    elif provider == "optimized":
        model = _opt_mod.ModelNew(*init_args)
    else:
        raise KeyError(provider)
    model = model.to(device="npu").eval()
    _MODEL_CACHE[key] = model
    return model


def _make_inputs(batch, in_features):
    _seed()
    return (torch.rand(batch, in_features, device="npu"),)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _baseline_grid_overflows(batch, groups):
    return batch * groups > 65535


def _run_provider(provider, batch, in_features, out_features, groups):
    init_args = _init_args(in_features, out_features, groups)
    if provider in ("baseline1", "baseline2") and _baseline_grid_overflows(batch, groups):
        raise RuntimeError("grid_guard")
    model = _model(provider, init_args)
    if model is None:
        raise RuntimeError("provider_unavailable")
    (x,) = _make_inputs(batch, in_features)
    with torch.no_grad():
        y = model(x)
    _sync()
    return y


def _compare(label, provider, batch, in_features, out_features, groups):
    if provider == "torch":
        _run_provider("torch", batch, in_features, out_features, groups)
        print(f"TEST torch {label} PASS shape=({batch},{in_features},{out_features},{groups})")
        return True
    try:
        ref = _run_provider("torch", batch, in_features, out_features, groups)
        out = _run_provider(provider, batch, in_features, out_features, groups)
        diff = (out - ref).abs()
        max_abs = diff.max().item()
        max_ref = ref.abs().max().item()
        ok_close = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
        if not ok_close:
            print(f"TEST {provider} {label} MISMATCH max_abs={max_abs:.6e} max_ref={max_ref:.6e} out0={out.flatten()[0].item():.6e} ref0={ref.flatten()[0].item():.6e}")
            return False
        print(f"TEST {provider} {label} PASS max_abs={max_abs:.6e}")
        return True
    except RuntimeError as exc:
        reason = str(exc)
        if provider != "optimized" and ("grid_guard" in reason or "provider_unavailable" in reason):
            print(f"TEST {provider} {label} SKIP {reason}")
            return True
        print(f"TEST {provider} {label} MISMATCH {type(exc).__name__}")
        return False
    except Exception as exc:
        if provider != "optimized":
            print(f"TEST {provider} {label} SKIP {type(exc).__name__}")
            return True
        print(f"TEST {provider} {label} MISMATCH {type(exc).__name__}")
        return False


def unit_test():
    ok = True
    for label, batch, in_features, out_features, groups in _BENCH_SHAPES:
        for provider in ("torch", "baseline1", "baseline2", "optimized"):
            ok = _compare(label, provider, batch, in_features, out_features, groups) and ok
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _manual_bench(fn, warmup=10, rep=50):
    with torch.no_grad():
        for _ in range(warmup):
            fn(); _sync()
        start = time.perf_counter()
        for _ in range(rep):
            fn(); _sync()
        end = time.perf_counter()
    return (end - start) * 1000.0 / rep


def _bench_provider(provider, batch, in_features, out_features, groups):
    if provider in ("baseline1", "baseline2") and _baseline_grid_overflows(batch, groups):
        print(f"INFO bench {provider} SKIP grid_guard")
        return float("inf")
    init_args = _init_args(in_features, out_features, groups)
    model = _model(provider, init_args)
    if model is None:
        print(f"INFO bench {provider} SKIP provider_unavailable")
        return float("inf")
    (x,) = _make_inputs(batch, in_features)

    def fn():
        with torch.no_grad():
            return model(x)

    try:
        if hasattr(triton, "testing") and hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="mean")
        return _manual_bench(fn)
    except Exception as exc:
        print(f"INFO bench {provider} INF {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "baseline1", "baseline2", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="matmul_swish_sum_groupnorm",
        args={},
    )
)
def benchmark(label, provider):
    batch, in_features, out_features, groups = _SHAPES[label]
    return _bench_provider(provider, batch, in_features, out_features, groups)


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
        benchmark.run(print_data=True, show_plots=False, save_path=None)


if __name__ == "__main__":
    main()
