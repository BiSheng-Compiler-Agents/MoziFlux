"""
profile_kernels.py — performance benchmark for l1_1 Square Matrix Multiplication
Compares baseline vs optimized kernel on the Ascend NPU.

Run on hardware with:
    python profile_kernels.py
"""
import os
import sys
import torch
import argparse

try:
    import torch_npu  # noqa: F401
    DEVICE = "npu"
except ImportError:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import triton
from triton.testing import do_bench, perf_report, Benchmark
from pathlib import Path
import importlib

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

_BENCH_SHAPES = [
    # (label,             n_elements)
    ("N=256",             256),          
    ("N=512",           512),           
    ("N=1024",         1024),      
    ("N=2048",           2048),   
    ("N=4096",         4096),      
]


configs = [
    Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="matmul_benchmark",
        args={},
    )
]

def _load(fname):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_baseline1  = _load(_DIR / "1_Square_matrix_multiplication_.py")
_baseline2  = _load(_DIR / "1_Square_matrix_multiplication_.py")
_optimized = _load(_DIR / "opt_1_Square_matrix_multiplication_.py")


@perf_report(configs)
def benchmark(label, provider):
    _, N = next(s for s in _BENCH_SHAPES if s[0] == label)
    A = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
    B = torch.randn(N, N, device=DEVICE, dtype=torch.float32)

    if provider == "torch_ref":
        return do_bench(lambda: torch.matmul(A, B))
    elif provider == "baseline1":
        m = _baseline1.ModelNew()
        return do_bench(lambda: m.forward(A, B))
    elif provider == "baseline2":
        m = _baseline2.ModelNew()
        return do_bench(lambda: m.forward(A, B))
    elif provider == "optimized":
        m = _optimized.ModelNew()
        return do_bench(lambda: m.forward(A, B))


def unit_test():
    torch.manual_seed(42)
    print("Correctness check (atol=1e-3, rtol=1e-3):")
    any_fail = False
    for label, N_dim in _BENCH_SHAPES:
        A = torch.randn(N_dim, N_dim, device=DEVICE, dtype=torch.float32)
        B = torch.randn(N_dim, N_dim, device=DEVICE, dtype=torch.float32)
        ref  = torch.matmul(A, B)
        base1 = _baseline1.ModelNew()(A.clone(), B.clone())
        base2 = _baseline2.ModelNew()(A.clone(), B.clone())
        opt  = _optimized.ModelNew()(A.clone(), B.clone())
        ok_b1 = torch.allclose(ref, base1, atol=1e-3, rtol=1e-3)
        ok_b2 = torch.allclose(ref, base2, atol=1e-3, rtol=1e-3)
        ok_o = torch.allclose(ref, opt,  atol=1e-3, rtol=1e-3)
        print(f"  {label:<26}  baseline1 [{'PASS' if ok_b1 else 'FAIL'}]  "
              f"optimized [{'PASS' if ok_b2 else 'FAIL'}]  "
              f"optimized [{'PASS' if ok_o else 'FAIL'}]  "
              f"maxDelta_base1={(ref-base1).abs().max():.2e}  "
              f"maxDelta_base2={(ref-base2).abs().max():.2e}  "
              f"maxDelta_opt={(ref-opt).abs().max():.2e}")
        if not ok_b1 or not ok_b2 or not ok_o:
            any_fail = True
    if any_fail:
        print("FAILED")
        sys.exit(1)
    print("All PASS")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="l1_1 matmul profiler")
    parser.add_argument("--test",  action="store_true", help="Correctness check only")
    parser.add_argument("--bench", action="store_true", help="Benchmark only")
    args = parser.parse_args()

    run_test  = args.test  or not args.bench
    run_bench = args.bench or not args.test

    if run_test:
        unit_test()
    if run_bench:
        benchmark.run(save_path=str(_DIR), print_data=True)


if __name__ == "__main__":
    main()