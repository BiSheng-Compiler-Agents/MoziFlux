"""
profile_kernels.py -- l1_2 Standard Matrix Multiplication
==========================================================
Three-way latency comparison on Ascend NPU hardware:
  torch_ref  : PyTorch / ACL   (ACL / cuBLAS built-in)
  baseline   : original Triton kernel  (2_Standard_matrix_multiplication_.py)
  optimized  : opt_2_Standard_matrix_multiplication_.py

Shapes tested:
  - Small square:  256x256x256    (dispatch overhead dominates)
  - Medium square: 512x512x512
  - Large square:  1024x1024x1024
  - Benchmark:     1024x4096x2048 (M=1024, K=4096, N=2048)
  - Non-square:    512x2048x1024
  - Large non-sq:  2048x1024x4096

Usage:
    python profile_kernels.py           # unit test + benchmark + saved figure
    python profile_kernels.py --test    # correctness check only
    python profile_kernels.py --bench   # benchmark only
"""
import argparse
import sys
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
    DEVICE = "npu"
except ImportError:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from triton.testing import do_bench, perf_report, Benchmark

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

import importlib.util  # noqa: E402


def _load(fname):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_baseline1 = _load(_DIR / "2_Standard_matrix_multiplication_.py")
_baseline2 = _load(_DIR / "base_2_Standard_matrix_multiplication_.py")
_optimized = _load(_DIR / "opt_2_Standard_matrix_multiplication_.py")

# ── shapes: (label, M, K, N) ──────────────────────────────────────────────────

_BENCH_SHAPES = [
    ("M256-K256-N256", 256, 256, 256),  # small square
    ("M512-K512-N512", 512, 512, 512),  # medium square
    ("M1024-K1024-N1024", 1024, 1024, 1024),  # large square
    ("M1024-K4096-N2048", 1024, 4096, 2048),  # benchmark shape
    ("M512-K2048-N1024", 512, 2048, 1024),  # tall K
    ("M2048-K1024-N4096", 2048, 1024, 4096),  # wide N
]

# ── runner functions ───────────────────────────────────────────────────────────


def _run_torch_ref(A, B):
    return torch.matmul(A, B)


def _run_baseline1(A, B):
    # Baseline kernel is wrapped in @triton.autotune, so BLOCK_M/BLOCK_N/BLOCK_K
    # are auto-tuned meta-parameters. They MUST NOT be passed as kwargs here —
    # doing so raises "Conflicting meta-parameters". Use the kernel's ModelNew
    # front-end, which lets the autotune wrapper pick the tiles.
    return _baseline1.ModelNew().forward(A, B)


def _run_baseline2(A, B):
    # Baseline kernel is wrapped in @triton.autotune, so BLOCK_M/BLOCK_N/BLOCK_K
    # are auto-tuned meta-parameters. They MUST NOT be passed as kwargs here —
    # doing so raises "Conflicting meta-parameters". Use the kernel's ModelNew
    # front-end, which lets the autotune wrapper pick the tiles.
    return _baseline2.ModelNew().forward(A, B)


def _run_optimized(A, B):
    m = _optimized.ModelNew()
    return m.forward(A, B)


# ── benchmark ──────────────────────────────────────────────────────────────────


@perf_report([
    Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="matmul_l1_2_perf",
        args={},
    )
])
def benchmark(label, provider):
    _, M_dim, K_dim, N_dim = next(s for s in _BENCH_SHAPES if s[0] == label)
    A = torch.randn(M_dim, K_dim, device=DEVICE, dtype=torch.float32)
    B = torch.randn(K_dim, N_dim, device=DEVICE, dtype=torch.float32)

    if provider == "torch_ref":

        def fn():
            return _run_torch_ref(A, B)
    elif provider == "baseline1":

        def fn():
            return _run_baseline1(A, B)
    elif provider == "baseline2":

        def fn():
            return _run_baseline2(A, B)
    else:

        def fn():
            return _run_optimized(A, B)

    return do_bench(fn, warmup=25, rep=200, return_mode="mean")


# ── unit test ──────────────────────────────────────────────────────────────────


def unit_test():
    torch.manual_seed(42)
    print("Correctness check (atol=1e-3, rtol=1e-3):")
    any_fail = False
    for label, M_dim, K_dim, N_dim in _BENCH_SHAPES:
        A = torch.randn(M_dim, K_dim, device=DEVICE, dtype=torch.float32)
        B = torch.randn(K_dim, N_dim, device=DEVICE, dtype=torch.float32)
        ref = _run_torch_ref(A, B)
        base1 = _run_baseline1(A.clone(), B.clone())
        base2 = _run_baseline2(A.clone(), B.clone())
        opt = _run_optimized(A.clone(), B.clone())
        ok_b1 = torch.allclose(ref, base1, atol=1e-3, rtol=1e-3)
        ok_b2 = torch.allclose(ref, base2, atol=1e-3, rtol=1e-3)
        ok_o = torch.allclose(ref, opt, atol=1e-3, rtol=1e-3)
        print(f"  {label:<26}  baseline1 [{'PASS' if ok_b1 else 'FAIL'}]  "
              f"baseline2 [{'PASS' if ok_b2 else 'FAIL'}]  "
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
    parser = argparse.ArgumentParser(description="l1_2 matmul profiler")
    parser.add_argument("--test",
                        action="store_true",
                        help="Correctness check only")
    parser.add_argument("--bench", action="store_true", help="Benchmark only")
    args = parser.parse_args()

    run_test = args.test or not args.bench
    run_bench = args.bench or not args.test

    if run_test:
        unit_test()
    if run_bench:
        benchmark.run(save_path=str(_DIR), print_data=True)


if __name__ == "__main__":
    main()
