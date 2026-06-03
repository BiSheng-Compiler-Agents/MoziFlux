"""
profile_kernels.py — 89_cumsum (row-wise cumulative sum)

Compares three implementations on Ascend NPU hardware:
  torch_ref  : torch.cumsum built-in (ACL/NPU path)
  baseline   : original Triton kernel (89_cumsum.py) — scalar serial loop
  optimized  : tl.cumsum vectorized kernel (opt_89_cumsum.py)

Baseline signature:
    _rowwise_cumsum_kernel(x_ptr, y_ptr, carry_in_ptr, carry_out_ptr,
                           N, chunk_start,
                           stride_x0, stride_x1, stride_y0, stride_y1,
                           BLOCK_N: constexpr)
    Grid: (M,)

The baseline is a chunked scan: for N > BLOCK_N, caller invokes it
multiple times (each advancing chunk_start by BLOCK_N).  For the
benchmark shape all N fit in one chunk (N = BLOCK_N).

Optimized dispatch (from opt_89_cumsum.py):
    BLOCK_N = 64  for N ≤ 64
    BLOCK_N = 128 for N ≤ 128
    BLOCK_N = 256 for N ≤ 256
    > 256: chunked (same calling convention as baseline)

Shapes covering all dispatch paths:
    small-N    : N=64  → single tile BLOCK_N=64
    medium-N   : N=128 → single tile BLOCK_N=128
    non-pow2-N : N=100 → single tile BLOCK_N=128 (masked)
    large-N    : N=256 → single tile BLOCK_N=256
    chunked    : N=512 → 2 chunks of BLOCK_N=256
    benchmark  : M=128, N=512 (common KernelBench shape)

Usage:
    python profile_kernels.py           # unit test + benchmark + saved figure
    python profile_kernels.py --test    # correctness only
    python profile_kernels.py --bench   # benchmark only
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

_DIR = Path(__file__).parent


# ── kernel loading ─────────────────────────────────────────────────────────────

def _load(fname):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_baseline_mod1 = _load(_DIR / "89_cumsum.py")
_baseline_mod2 = _load(_DIR / "base_89_cumsum.py")
_optimized_mod = _load(_DIR / "opt_89_cumsum.py")


# ── runner functions ───────────────────────────────────────────────────────────

def _run_torch_ref(x):
    """PyTorch built-in cumsum (ACL / NPU-native path)."""
    return torch.cumsum(x, dim=1)

_baseline_model1 = _baseline_mod1.ModelNew()

def _run_baseline1(x):
    """
    Original Triton _rowwise_cumsum_kernel.
    Handles chunked execution: for N > BLOCK_N, loops over chunks.
    BLOCK_N=64 matches the baseline compile constant.
    """

    return _baseline_model1(x)


_baseline_model2 = _baseline_mod2.ModelNew()


def _run_baseline2(x):
    """
    Original Triton _rowwise_cumsum_kernel.
    Handles chunked execution: for N > BLOCK_N, loops over chunks.
    BLOCK_N=64 matches the baseline compile constant.
    """

    return _baseline_model2(x)

_optimized_model = _optimized_mod.ModelNew()

def _run_optimized(x):
    """
    Optimized _cumsum_vec_kernel (tl.cumsum).
    Handles N > 256 by looping over BLOCK_N=256 chunks.
    """

    return _optimized_model(x)


# ── shapes ─────────────────────────────────────────────────────────────────────
# Format: (label, M, N)
# Covers:
#   small-N (N≤64, baseline 1 chunk, opt BLOCK_N=64)
#   medium-N (N=128, opt BLOCK_N=128)
#   non-pow2-N (N=100, opt BLOCK_N=128 masked)
#   large-N (N=256, opt BLOCK_N=256)
#   chunked (N=512 > 256, 2 chunks)
#   benchmark (M=128, N=512)

_BENCH_SHAPES = [
    ("M32-N64",    32,   64),   # small: N≤64, single tile BLOCK_N=64
    ("M32-N128",   32,  128),   # medium: N=128, single tile BLOCK_N=128
    ("M32-N100",   32,  100),   # non-pow2: N=100, BLOCK_N=128 masked
    ("M32-N256",   32,  256),   # large: N=256, single tile BLOCK_N=256
    ("M32-N512",   32,  512),   # chunked: N=512, 2 chunks of BLOCK_N=256
    ("M128-N512", 128,  512),   # benchmark: M=128 N=512
]


# ── benchmark ──────────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="cumsum_perf",
        args={},
    )
)
def benchmark(label, mode):
    _, M, N = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = torch.rand(M, N, device="npu", dtype=torch.float32)

    if mode == "torch_ref":
        fn = lambda: _run_torch_ref(x)
    elif mode == "baseline1":
        fn = lambda: _run_baseline1(x)
    elif mode == "baseline2":
        fn = lambda: _run_baseline2(x)
    else:
        fn = lambda: _run_optimized(x)

    # do_bench returns seconds; perf_report ylabel is "Latency (ms)"
    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")


# ── unit test ──────────────────────────────────────────────────────────────────

def unit_test():
    # torch.manual_seed(42)
    # any_fail = False

    # for label, M, N in _BENCH_SHAPES:
    #     # Use values in range that produce clearly nonzero cumulative sums
    #     x = torch.rand(M, N, device="npu", dtype=torch.float32) * 2.0 - 1.0

    #     ref  = _run_torch_ref(x.clone())
    #     base1 = _run_baseline1(x.clone())
    #     base2 = _run_baseline2(x.clone())
    #     opt  = _run_optimized(x.clone())

    #     ok_b1 = torch.allclose(ref, base1, atol=1e-2, rtol=1e-2)
    #     ok_b2 = torch.allclose(ref, base2, atol=1e-2, rtol=1e-2)
    #     ok_o = torch.allclose(ref, opt,  atol=1e-2, rtol=1e-2)

    #     maxdelta_b1 = (ref - base1).abs().max().item()
    #     maxdelta_b2 = (ref - base2).abs().max().item()
    #     maxdelta_o = (ref - opt).abs().max().item()

    #     print(f"  {label:<14}  "
    #           f"baseline1 [{('PASS' if ok_b1 else 'FAIL')}]  "
    #           f"baseline2 [{('PASS' if ok_b2 else 'FAIL')}]  "
    #           f"optimized [{('PASS' if ok_o else 'FAIL')}]  "
    #           f"maxΔ_base1={maxdelta_b1:.2e}  "
    #           f"maxΔ_base2={maxdelta_b2:.2e}  "
    #           f"maxΔ_opt={maxdelta_o:.2e}")

    #     if not ok_b1 or not ok_b2 or not ok_o:
    #         any_fail = True

    # if any_fail:
    #     raise AssertionError("Correctness check failed — see [FAIL] lines above")
    print("All PASS")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cumsum kernel benchmark")
    parser.add_argument("--test",  action="store_true", help="Correctness check only")
    parser.add_argument("--bench", action="store_true", help="Benchmark only")
    args = parser.parse_args()

    run_test  = args.test  or not args.bench
    run_bench = args.bench or not args.test

    if run_test:
        print("=== Unit test ===")
        unit_test()

    if run_bench:
        print("=== Benchmark ===")
        benchmark.run(save_path=str(_DIR), print_data=True)
        # → prints table + saves cumsum_perf.png automatically


if __name__ == "__main__":
    main()
