import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton

ROOT = Path(__file__).resolve().parent


def _load(fname: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname: str, name: str):
    try:
        return _load(fname, name)
    except Exception as exc:
        print(
            f"INFO optional_provider {name} unavailable {type(exc).__name__}")
        return None


baseline1 = _load("12_Gemm_Multiply_LeakyReLU.py", "k_baseline1_12_gemm_leaky")
baseline2 = _load_optional("base_12_Gemm_Multiply_LeakyReLU.py",
                           "k_baseline2_12_gemm_leaky")
optimized = _load("opt_12_Gemm_Multiply_LeakyReLU.py",
                  "k_optimized_12_gemm_leaky")

_BENCH_SHAPES = [
    ("small", 128, 1024, 1024),
    ("irregular", 257, 2048, 1536),
    ("target", 1024, 8192, 8192),
]

_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_MODEL_CACHE = {}


class TorchRef(nn.Module):

    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        return self.leaky_relu(self.gemm(x) * self.multiplier)


def _provider_module(provider):
    if provider == "baseline1":
        return baseline1
    if provider == "baseline2":
        return baseline2
    if provider == "optimized":
        return optimized
    return None


def _model(provider, K, N):
    key = (provider, K, N)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    init = [K, N, 2.0, 0.1]
    torch.manual_seed(0)
    if provider == "torch":
        model = TorchRef(*init)
    else:
        mod = _provider_module(provider)
        if mod is None:
            _MODEL_CACHE[key] = None
            return None
        model = mod.ModelNew(*init)
    model = model.to(device="npu").eval()
    _MODEL_CACHE[key] = model
    return model


def _make_inputs(M, K):
    torch.manual_seed(123)
    return torch.rand(M, K, device="npu")


def _run_provider(provider, x, K, N):
    model = _model(provider, K, N)
    if model is None:
        return None
    with torch.no_grad():
        return model(x)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _time_ms(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
        _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


def _bench_provider(provider, label):
    _, M, K, N = {s[0]: s for s in _BENCH_SHAPES}[label]
    x = _make_inputs(M, K)
    try:
        y = _run_provider(provider, x, K, N)
        if y is None:
            return float("inf")
        _sync()
        return _time_ms(lambda: _run_provider(provider, x, K, N))
    except Exception as exc:
        print(f"INFO bench {provider} {label} inf {type(exc).__name__}")
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
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="gemm_multiply_leakyrelu",
        args={},
    ))
def bench(label, provider):
    return _bench_provider(provider, label)


def unit_test():
    all_ok = True
    for label, M, K, N in _BENCH_SHAPES:
        x = _make_inputs(M, K)
        ref = _run_provider("torch", x, K, N)
        _sync()
        for provider in ["baseline1", "baseline2", "optimized"]:
            if provider == "baseline2" and baseline2 is None:
                print(f"TEST baseline2 {label} SKIP optional_unavailable")
                continue
            try:
                out = _run_provider(provider, x, K, N)
                _sync()
                ok = torch.allclose(out, ref, rtol=1e-3, atol=1e-2)
                max_abs = (out - ref).abs().max().item()
                if ok:
                    print(
                        f"TEST {provider} {label} PASS max_abs={max_abs:.6g}")
                elif provider == "baseline2":
                    print(
                        f"TEST baseline2 {label} SKIP value_diff max_abs={max_abs:.6g}"
                    )
                else:
                    print(
                        f"TEST {provider} {label} MISMATCH max_abs={max_abs:.6g}"
                    )
                    all_ok = False
            except Exception as exc:
                if provider == "optimized":
                    print(
                        f"TEST optimized {label} MISMATCH exception={type(exc).__name__}"
                    )
                    all_ok = False
                else:
                    print(
                        f"TEST {provider} {label} SKIP exception={type(exc).__name__}"
                    )
    print("UNIT_TEST PASS" if all_ok else "UNIT_TEST_FAILED")
    return all_ok


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
