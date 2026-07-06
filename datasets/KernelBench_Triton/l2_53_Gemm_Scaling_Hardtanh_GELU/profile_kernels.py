import argparse
import importlib.util
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = pathlib.Path(__file__).resolve().parent
INPUT_FILE = ROOT / "53_Gemm_Scaling_Hardtanh_GELU.py"
BASE_FILE = ROOT / "base_53_Gemm_Scaling_Hardtanh_GELU.py"
OPT_FILE = ROOT / "opt_53_Gemm_Scaling_Hardtanh_GELU.py"

_BENCH_SHAPES = [
    ("small", 256, 1024, 512),
    ("irregular", 257, 1152, 768),
    ("target", 2048, 8192, 8192),
]
_PROVIDERS = ["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"]
_MODULE_CACHE = {}
_MODEL_CACHE = {}


def _load(path, key):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def _init_args(m, k, n):
    return [k, n, 0.5, -2.0, 2.0]


class TorchRef(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor=0.5, hardtanh_min=-2.0, hardtanh_max=2.0):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        y = self.gemm(x)
        y = torch.clamp(y * self.scaling_factor, self.hardtanh_min, self.hardtanh_max)
        return F.gelu(y, approximate="none")


def _model(provider, m, k, n):
    cache_key = (provider, m, k, n)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    torch.manual_seed(0)
    if provider == "PyTorch / ACL":
        model = TorchRef(*_init_args(m, k, n))
    elif provider == "Baseline Triton1":
        mod = _load(INPUT_FILE, "baseline1")
        model = mod.ModelNew(*_init_args(m, k, n))
    elif provider == "Baseline Triton2":
        mod = _load(BASE_FILE, "baseline2")
        model = mod.ModelNew(*_init_args(m, k, n))
    elif provider == "Optimized Triton":
        mod = _load(OPT_FILE, "optimized")
        model = mod.ModelNew(*_init_args(m, k, n))
    else:
        raise ValueError(provider)
    model = model.eval().npu()
    _MODEL_CACHE[cache_key] = model
    return model


def _make_inputs(m, k, n):
    torch.manual_seed(123)
    return torch.rand((m, k), device="npu", dtype=torch.float32)


def _run_provider(provider, label):
    _, m, k, n = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = _make_inputs(m, k, n)
    with torch.no_grad():
        return _model(provider, m, k, n)(x)


def _max_diff(a, b):
    return float((a.float() - b.float()).abs().max().detach().cpu())


def unit_test():
    ok = True
    for label, m, k, n in _BENCH_SHAPES:
        ref = _run_provider("PyTorch / ACL", label)
        for provider in _PROVIDERS[1:]:
            try:
                out = _run_provider(provider, label)
                diff = _max_diff(out, ref)
                passed = math.isfinite(diff) and diff <= 1e-3
                print(f"CHECK {provider} {label}: max_abs_diff={diff:.6g} {'PASS' if passed else 'MISMATCH'}")
                if provider == "Optimized Triton" and not passed:
                    ok = False
            except Exception as exc:
                print(f"INFO {provider} {label}: unavailable_or_preskipped {type(exc).__name__}")
                if provider == "Optimized Triton":
                    ok = False
        if label == "small":
            try:
                opt = _load(OPT_FILE, "optimized")
                old = getattr(opt, "_MAX_PROGRAMS")
                opt._MAX_PROGRAMS = 1
                _MODEL_CACHE.pop(("Optimized Triton", m, k, n), None)
                out = _run_provider("Optimized Triton", label)
                diff = _max_diff(out, ref)
                passed = math.isfinite(diff) and diff <= 1e-3
                print(f"CHECK Optimized Triton forced_persistent {label}: max_abs_diff={diff:.6g} {'PASS' if passed else 'MISMATCH'}")
                ok = ok and passed
            finally:
                opt._MAX_PROGRAMS = old
                _MODEL_CACHE.pop(("Optimized Triton", m, k, n), None)
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_one(provider, label):
    try:
        fn = lambda: _run_provider(provider, label)
        if hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn, warmup=5, rep=20, return_mode="mean")
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / 20.0
    except Exception as exc:
        print(f"INFO bench {provider} {label}: inf {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=_PROVIDERS,
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="gemm_scaling_hardtanh_gelu",
        args={},
    )
)
def benchmark(label, provider):
    return _bench_one(provider, label)


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
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
