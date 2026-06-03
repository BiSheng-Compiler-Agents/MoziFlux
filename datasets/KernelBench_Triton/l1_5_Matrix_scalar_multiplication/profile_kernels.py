"""
profile_kernels.py — l1_5 Matrix × Scalar Multiplication

Compares three implementations on Ascend NPU hardware:
  torch_ref  : PyTorch / ACL built-in path  (x * scalar)
  baseline   : original Triton kernel from 5_Matrix_scalar_multiplication.py
  optimized  : trace-optimized kernel from opt_5_Matrix_scalar_multiplication.py
               (ModelNew with two-path dispatch: direct for n_tiles ≤ 65535,
                persistent for n_tiles > 65535)

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
import triton.testing

_DIR = Path(__file__).parent

# ── Load baseline + optimized via importlib (never copy kernel code) ──────────
def _load(fname: Path):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_baseline_mod1 = _load(_DIR / "5_Matrix_scalar_multiplication.py")
_baseline_mod2 = _load(_DIR / "base_5_Matrix_scalar_multiplication.py")
_optimized_mod = _load(_DIR / "opt_5_Matrix_scalar_multiplication.py")

# ── runner functions ──────────────────────────────────────────────────────────
SCALAR = 3.14


def _run_torch_ref(x: torch.Tensor) -> torch.Tensor:
    return x * SCALAR


def _run_baseline1(x: torch.Tensor) -> torch.Tensor:
    """Local baseline ModelNew (forward(A, s) signature, BLOCK_SIZE=16384).

    The baseline kernel has no FFTS guard — at n_tiles > 65535 it would crash
    with `coreDim > UINT16_MAX`. We chunk along the flattened tensor so the
    baseline runs in ≤ 65535-tile sub-grids, matching the optimized kernel's
    routing threshold. This guard is part of the benchmark harness, not the
    baseline kernel itself.
    """
    BLOCK = 16384
    n_elements = x.numel()
    n_tiles = triton.cdiv(n_elements, BLOCK)
    if n_tiles <= 65535:
        return _baseline_mod1.ModelNew()(x, float(SCALAR))
    # Chunk: split into pieces of ≤ 65535 tiles each
    chunk_n = 65535 * BLOCK
    out = torch.empty_like(x)
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    model = _baseline_mod1.ModelNew()
    for start in range(0, n_elements, chunk_n):
        end = min(start + chunk_n, n_elements)
        chunk_x = x_flat[start:end].view(-1)
        chunk_out = model(chunk_x, float(SCALAR))
        out_flat[start:end].copy_(chunk_out.view(-1))
    return out

def _run_baseline2(x: torch.Tensor) -> torch.Tensor:
    """ModelNew with two-path dispatch (direct or persistent based on n_tiles)."""
    model = _baseline_mod2.ModelNew()
    return model(x, SCALAR)


def _run_optimized(x: torch.Tensor) -> torch.Tensor:
    """ModelNew with two-path dispatch (direct or persistent based on n_tiles)."""
    model = _optimized_mod.ModelNew(SCALAR)
    return model(x)


# ── shape table — covers all dispatch paths ───────────────────────────────────
# Format: (label, M, N) — total elements drives which path the optimized kernel
# takes and what the baseline must handle.
_BENCH_SHAPES = [
    ("tiny-32x32",         32,    32),    # n=1024      → direct (small)
    ("small-256x256",      256,   256),   # n=65K       → direct
    ("medium-1kx1k",       1024,  1024),  # n=1M        → direct
    ("bench-4kx4k",        4096,  4096),  # n=16M       → direct (n_tiles=4096)
    ("large-8kx8k",        8192,  8192),  # n=64M       → direct
    ("huge-32kx32k",       32768, 32768), # n=1G        → persistent (n_tiles=262144)
    ("rect-2kx16k",        2048,  16384), # n=32M       → direct (rectangular)
]


# ── benchmark ────────────────────────────────────────────────────────────────
configs = [
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="l1_5_matrix_scalar_multiplication",
        args={},
    )
]


@triton.testing.perf_report(configs)
def benchmark(label, mode):
    _, M, N = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = torch.rand(M, N, dtype=torch.float32, device="npu")
    fn = (_run_torch_ref if mode == "torch_ref"
          else _run_baseline1 if mode == "baseline1"
          else _run_baseline2 if mode == "baseline2"
          else _run_optimized)
    return triton.testing.do_bench(lambda: fn(x), warmup=25, rep=200, return_mode="mean")


# ── unit test ────────────────────────────────────────────────────────────────
# All shapes are tested for both correctness and that every dispatch path
# (direct / persistent) executes without error. Boundaries: empty tensor,
# n=1, non-power-of-2, bench shape.

def unit_test():
    torch.manual_seed(42)
    any_fail = False

    test_shapes = [
        # (label, M, N)
        ("1x1",            1,       1),
        ("16",             16,      1),         # n=16
        ("4095",           4095,    1),         # non-pow2 boundary
        ("4096",           4096,    1),         # pow2 boundary
        ("4097",           4097,    1),         # non-pow2 boundary
        ("1024x1024",      1024,    1024),
        ("bench-4096x4096",4096,    4096),
        ("8192x8192",      8192,    8192),
        ("rect-2kx16k",    2048,    16384),
        # Persistent path tests (n_tiles > 65535):
        #   32768x32768 = 1G elements, BLOCK=4096, n_tiles=262144
        ("persistent-32kx32k", 32768, 32768),
    ]

    for label, M, N in test_shapes:
        x = torch.rand(M, N, dtype=torch.float32, device="npu")
        ref  = _run_torch_ref(x)
        base1 = _run_baseline1(x)
        base2 = _run_baseline2(x)
        opt  = _run_optimized(x)
        ok_b1 = torch.allclose(ref, base1, atol=1e-4, rtol=1e-4)
        ok_b2 = torch.allclose(ref, base2, atol=1e-4, rtol=1e-4)
        ok_o = torch.allclose(ref, opt,  atol=1e-4, rtol=1e-4)
        max_d_b1 = (ref - base1).abs().max().item()
        max_d_b2 = (ref - base2).abs().max().item()
        max_d_o = (ref - opt).abs().max().item()
        print(f"  {label:<24}  baseline1 [{'PASS' if ok_b1 else 'FAIL'}]  "
              f"baseline2 [{'PASS' if ok_b2 else 'FAIL'}]  "
              f"optimized [{'PASS' if ok_o else 'FAIL'}]  "
              f"maxΔ_base1={max_d_b1:.2e} maxΔ_base2={max_d_b2:.2e} maxΔ_opt={max_d_o:.2e}")
        if not ok_b1 or not ok_b2 or not ok_o:
            any_fail = True

    # Empty-tensor edge case (must return empty, not crash)
    empty = torch.empty(0, dtype=torch.float32, device="npu")
    out = _run_optimized(empty)
    assert out.numel() == 0, "empty input produced non-empty output"
    print("  empty-tensor                          optimized [PASS]")

    if any_fail:
        raise AssertionError("Correctness check failed")
    print("All PASS")


# ── entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
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
