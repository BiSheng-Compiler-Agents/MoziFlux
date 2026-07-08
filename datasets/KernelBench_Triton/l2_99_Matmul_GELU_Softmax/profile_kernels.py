import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parent

_BENCH_SHAPES = [
    ("small_16x512x512", 16, 512, 512),
    ("medium_128x2048x2048", 128, 2048, 2048),
    ("target_1024x8192x8192", 1024, 8192, 8192),
]

PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
PROVIDER_NAMES = [
    "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
]
_MODEL_CACHE = {}
_MOD_CACHE = {}
_REF_CACHE = {}


def _load(key, filename):
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MOD_CACHE[key] = mod
    return mod


def _sync():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _device():
    return "npu" if hasattr(torch,
                            "npu") and torch.npu.is_available() else "cuda"


def _make_input(B, K, dtype=torch.float16):
    torch.manual_seed(123)
    return torch.rand((B, K), device=_device(), dtype=dtype)


def _init_args_from_shape(B, K, N):
    return [K, N]


def _model(key, B, K, N):
    cache_key = (key, K, N)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    filename = {
        "baseline1": "99_Matmul_GELU_Softmax.py",
        "baseline2": "base_99_Matmul_GELU_Softmax.py",
        "optimized": "opt_99_Matmul_GELU_Softmax.py",
    }[key]
    mod = _load(key, filename)
    torch.manual_seed(0)
    init = _init_args_from_shape(B, K, N)
    if init == [()]:
        init = []
    m = mod.ModelNew(*init).to(_device()).eval()
    _MODEL_CACHE[cache_key] = m
    return m


def _ref_model(B, K, N):
    cache_key = (K, N)
    if cache_key in _REF_CACHE:
        return _REF_CACHE[cache_key]
    torch.manual_seed(0)
    m = nn.Linear(K, N).to(_device()).eval().half()
    _REF_CACHE[cache_key] = m
    return m


def _run_torch_ref(x, B, K, N):
    m = _ref_model(B, K, N)
    if m.weight.dtype != x.dtype:
        m = m.to(dtype=x.dtype)
        _REF_CACHE[(K, N)] = m
    return torch.softmax(F.gelu(F.linear(x, m.weight, m.bias)), dim=-1)


def _run_provider(provider, x, B, K, N):
    if provider == "torch":
        return _run_torch_ref(x, B, K, N)
    m = _model(provider, B, K, N)
    if next(m.parameters()).dtype != x.dtype:
        m = m.to(dtype=x.dtype)
    return m(x)


def _comparison_preskip(provider, B, K, N):
    # Original row-wise Triton baselines compute GEMM with vector reductions and are
    # prohibitively slow at medium/target GEMM sizes; keep their columns as inf.
    return provider in {"baseline1", "baseline2"} and (B * K * N
                                                       > 16 * 512 * 512)


def _check_one(provider, label, B, K, N):
    if _comparison_preskip(provider, B, K, N):
        print(
            f"INFO {provider} {label} preskip: comparison baseline is too slow for this GEMM size"
        )
        return True
    x = _make_input(B, K, dtype=torch.float16)
    with torch.no_grad():
        ref = _run_torch_ref(x, B, K, N)
        _sync()
        try:
            out = _run_provider(provider, x, B, K, N)
            _sync()
        except Exception as exc:
            print(
                f"INFO {provider} {label} exception: {type(exc).__name__}: {exc}"
            )
            return provider in {"baseline2"}
    atol, rtol = 2e-2, 2e-2
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    if not ok:
        diff = (out - ref).abs()
        print(
            f"FAIL {provider} {label}: max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}"
        )
        return False
    print(f"PASS {provider} {label}")
    return True


def unit_test():
    overall = True
    for label, B, K, N in _BENCH_SHAPES:
        for provider in ["baseline1", "baseline2", "optimized"]:
            ok = _check_one(provider, label, B, K, N)
            if provider == "optimized":
                overall = overall and ok
    print("UNIT_TEST PASS" if overall else "UNIT_TEST_FAILED")
    return overall


def _manual_bench(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


def bench_provider(provider, label):
    B, K, N = next(
        (b, k, n) for (lbl, b, k, n) in _BENCH_SHAPES if lbl == label)
    if _comparison_preskip(provider, B, K, N):
        print(
            f"INFO bench {provider} {label} preskip: comparison baseline is too slow for this GEMM size"
        )
        return float("inf")
    x = _make_input(B, K, dtype=torch.float16)
    try:
        _run_provider(provider, x, B, K, N)
        _sync()
    except Exception as exc:
        print(
            f"INFO bench {provider} {label} unavailable: {type(exc).__name__}: {exc}"
        )
        return float("inf")

    def fn():
        return _run_provider(provider, x, B, K, N)

    try:
        return triton.testing.do_bench(fn,
                                       warmup=25,
                                       rep=100,
                                       return_mode="mean")
    except Exception as exc:
        print(
            f"INFO do_bench fallback {provider} {label}: {type(exc).__name__}: {exc}"
        )
        try:
            return _manual_bench(fn)
        except Exception as exc2:
            print(
                f"INFO bench {provider} {label} failed: {type(exc2).__name__}: {exc2}"
            )
            return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="Latency (ms)",
        plot_name="matmul_gelu_softmax",
        args={},
    ))
def benchmark(label, provider):
    return bench_provider(provider, label)


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
        benchmark.run(print_data=True,
                      show_plots=False,
                      save_path=str(ROOT / "profile_plots"))


if __name__ == "__main__":
    main()
