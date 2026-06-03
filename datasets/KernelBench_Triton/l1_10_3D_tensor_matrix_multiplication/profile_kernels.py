"""
profile_kernels.py — Benchmark + unit test for 3D Tensor Matrix Multiplication.

Tests ModelNew (optimized) against torch matmul and baseline Triton for correctness.
Benchmarks across multiple shapes using @triton.testing.perf_report.

Runs on real Ascend NPU hardware — use remote_verify tool.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton
import triton.language as tl

_DIR = Path(__file__).parent


def _load(fname):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Load sibling kernel source files.
# baseline1 == input kernel, baseline2 == golden reference (base_*), optimized == opt_*
_baseline1 = _load(_DIR / "10_3D_tensor_matrix_multiplication.py")
_baseline2 = _load(_DIR / "base_10_3D_tensor_matrix_multiplication.py")
_optimized = _load(_DIR / "opt_10_3D_tensor_matrix_multiplication.py")


# ---------------------------------------------------------------------------
# Lazy NPU model init
# ---------------------------------------------------------------------------
_models = {}


def _model(key):
    if key not in _models:
        mod = {"baseline1": _baseline1, "baseline2": _baseline2, "optimized": _optimized}[key]
        torch.manual_seed(0)
        _models[key] = mod.ModelNew().to(device="npu", dtype=torch.float32).eval()
    return _models[key]


# Provider order: torch ref first, then the three Triton variants.
_PROVIDERS = [
    ("torch_ref",  "PyTorch / ACL"),
    ("baseline1",  "Baseline Triton1"),
    ("baseline2",  "Baseline Triton2"),
    ("optimized",  "Optimized Triton"),
]
_VARIANTS = [(k, n) for k, n in _PROVIDERS if k != "torch_ref"]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def _run_torch_ref(A, B):
    return torch.matmul(A.float(), B.float()).half()


def _run_provider(key, A, B):
    if key == "torch_ref":
        return _run_torch_ref(A, B)
    return _model(key)(A, B)


# ---------------------------------------------------------------------------
# Shape table
# ---------------------------------------------------------------------------
_BENCH_SHAPES = [
    # label                 B     M     N     K
    # NOTE: labels must contain NO spaces — generate_report.py's table parser
    # splits rows on whitespace and treats a spaced label as misaligned columns.
    ("B1-small",            1,   128,  128,  256),
    ("B4-small",            4,   128,  128,  256),
    ("B1-medium",           1,   512,  512,  512),
    ("B4-medium",           4,   512,  512,  512),
    ("B16-medium",         16,   256,  256,  256),
    ("B8-tallM",            8,  1024,  512,  256),
    ("B8-wideN",            8,   512, 1024,  256),
    ("B2-nonpow2",          2,   768,  768,  768),
    ("B4-nonpow2",          4,   640,  832,  576),
    ("B8-large",            8,   512,  512,  512),
    ("B1-benchmark",        1,  1024, 1024, 1024),
    ("B4-benchmark",        4,  1024, 1024, 1024),
]


# ---------------------------------------------------------------------------
# Unit test with detailed debugging
# ---------------------------------------------------------------------------
def unit_test():
    print("=" * 72)
    print("UNIT TEST — every variant vs torch.matmul")
    print("=" * 72)
    any_fail = False

    # Pre-warm
    torch.manual_seed(0)
    _run_torch_ref(
        torch.rand(1, 128, 256, device="npu", dtype=torch.float16),
        torch.rand(256, 128, device="npu", dtype=torch.float16),
    )

    for label, B, M, N, K in _BENCH_SHAPES:
        torch.manual_seed(42)
        A = torch.randn(B, M, K, device="npu", dtype=torch.float16)
        b_mat = torch.randn(K, N, device="npu", dtype=torch.float16)
        with torch.no_grad():
            ref = _run_torch_ref(A, b_mat)

        cells = []
        for key, name in _VARIANTS:
            try:
                with torch.no_grad():
                    out = _run_provider(key, A, b_mat)
                err = (out.float() - ref.float()).abs().max().item()
                ok = err < 1.0
            except Exception as e:
                ok, err = False, float("nan")
                cells.append(f"{name}=[ERROR:{type(e).__name__}: {str(e)[:50]}]")
                any_fail = True
                continue
            cells.append(f"{name}=[{'PASS' if ok else 'FAIL'} {err:.2e}]")
            if not ok:
                any_fail = True
        print(f"  [{label:<14}] " + "  ".join(cells))

    # ---- Broadcasting tests — every variant vs torch ref ----
    print()
    print("Broadcasting Tests:")
    bcast = [
        ("2Dx3D", torch.randn(256, 128, device="npu", dtype=torch.float16),
                  torch.randn(4, 128, 512, device="npu", dtype=torch.float16)),
        ("3Dx2D", torch.randn(4, 256, 128, device="npu", dtype=torch.float16),
                  torch.randn(128, 512, device="npu", dtype=torch.float16)),
        ("2Dx2D", torch.randn(256, 128, device="npu", dtype=torch.float16),
                  torch.randn(128, 512, device="npu", dtype=torch.float16)),
    ]
    for label, A, b_mat in bcast:
        with torch.no_grad():
            ref = torch.matmul(A.float(), b_mat.float()).half()
        cells = []
        for key, name in _VARIANTS:
            try:
                with torch.no_grad():
                    out = _run_provider(key, A, b_mat)
                err = (out.float() - ref.float()).abs().max().item()
                ok = err < 1.0
            except Exception as e:
                ok = False
                cells.append(f"{name}=[ERROR:{type(e).__name__}: {str(e)[:50]}]")
                any_fail = True
                continue
            cells.append(f"{name}=[{'PASS' if ok else 'FAIL'} {err:.2e}]")
            if not ok:
                any_fail = True
        print(f"  [{label:<14}] " + "  ".join(cells))

    print()
    print("Some tests FAILED!" if any_fail else "All correctness tests PASSED!")
    print()
    return not any_fail
# Benchmark — @perf_report
# ---------------------------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=[k for k, _ in _PROVIDERS],
        line_names=[n for _, n in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="3D_tensor_matmul_perf",
        args={},
    )
)
def benchmark(label, mode):
    _, B, M, N, K = next(s for s in _BENCH_SHAPES if s[0] == label)
    A = torch.randn(B, M, K, device="npu", dtype=torch.float16)
    b_mat = torch.randn(K, N, device="npu", dtype=torch.float16)
    try:
        return triton.testing.do_bench(
            lambda: _run_provider(mode, A, b_mat),
            warmup=25, rep=200, return_mode="mean",
        )
    except Exception as e:
        print(f"[BENCH] {mode} failed on {label}: {type(e).__name__}: {e}")
        return float("inf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true", help="Correctness check only")
    parser.add_argument("--bench", action="store_true", help="Benchmark only")
    args = parser.parse_args()

    run_test  = args.test  or not args.bench
    run_bench = args.bench or not args.test

    ok = True
    try:
        if run_test:
            ok = unit_test()
        if run_bench:
            try:
                benchmark.run(save_path=str(_DIR), print_data=True)
            except Exception as e:
                print(f"\n[BENCH] perf_report failed: {e}")
                print("[BENCH] Manual benchmark skipped (requires NPU do_bench support)")
    except Exception as e:
        print(f"\n[FATAL] {e}")
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()