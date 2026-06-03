"""profile_kernels.py — Benchmark and correctness test for l1_3 Batched Matrix Multiplication.

Correctness checks EVERY variant (baseline1, baseline2, optimized) against the
torch reference. The benchmark is resilient: a provider that fails on a shape
(e.g. autotune "No valid triton configs") yields inf for that cell instead of
aborting the whole run.

Usage:
    python profile_kernels.py            # correctness + benchmark
    python profile_kernels.py --test     # correctness only
    python profile_kernels.py --bench    # benchmark only
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))


def _load(fname):
    """Load a sibling kernel file as a module under a safe identifier.

    File stems like '3_Batched_...' start with a digit (invalid module name),
    so we prefix with 'k_' to keep importlib happy and avoid collisions.
    """
    path = _DIR / fname
    mod_name = "k_" + path.stem
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# baseline1 == the input kernel (<N>_<name>.py)
# baseline2 == the golden reference kernel (base_<N>_<name>.py)
# optimized == opt_<N>_<name>.py
_baseline1 = _load("3_Batched_matrix_multiplication.py")
_baseline2 = _load("base_3_Batched_matrix_multiplication.py")
_optimized = _load("opt_3_Batched_matrix_multiplication.py")

# Map provider key -> (display name, loader module or None for torch ref)
_PROVIDERS = [
    ("torch_ref",  "PyTorch / ACL",      None),
    ("baseline1",  "Baseline Triton1",   _baseline1),
    ("baseline2",  "Baseline Triton2",   _baseline2),
    ("optimized",  "Optimized Triton",   _optimized),
]


def _torch_ref(A, B):
    """Reference matmul. torch.matmul handles 2D/3D broadcasting natively, so
    it matches whatever broadcasting the kernels implement without manual
    unsqueeze/expand (which silently mismatched batch dims)."""
    out = torch.matmul(A.float(), B.float())
    return out.to(A.dtype)


def _run_provider(key, mod, A, B):
    """Run one provider. torch_ref uses the reference; others call ModelNew."""
    if key == "torch_ref":
        return _torch_ref(A, B)
    return mod.ModelNew()(A, B)


# --- Correctness: every variant vs torch reference ---------------------------

def test_correctness():
    if not torch.npu.is_available():
        print("[TEST] NPU not available — skipping hardware test")
        return True

    test_cases = [
        # (batch, M, K, N, dtype, label)
        (2, 32, 64, 16,     torch.float16,  "small_FP16"),
        (2, 128, 128, 128,  torch.float16,  "square_FP16"),
        (4, 256, 128, 64,   torch.float16,  "tall_FP16"),
        (4, 64, 128, 256,   torch.float16,  "wide_FP16"),
        (2, 128, 64, 128,   torch.bfloat16, "BF16"),
        (2, 64, 64, 64,     torch.float32,  "FP32"),
        (1, 128, 256, 128,  torch.float16,  "batch1_FP16"),
        (8, 512, 1024, 2048,torch.float16,  "large_FP16"),
    ]

    variants = [(k, name, mod) for k, name, mod in _PROVIDERS if k != "torch_ref"]
    all_pass = True

    for batch, M, K, N, dtype, label in test_cases:
        torch.manual_seed(42)
        A = torch.randn(batch, M, K, device="npu", dtype=dtype)
        B = torch.randn(batch, K, N, device="npu", dtype=dtype)
        atol, rtol = (1e-2, 1e-2) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-4)

        with torch.no_grad():
            ref = _torch_ref(A, B)

        cells = []
        for key, name, mod in variants:
            try:
                with torch.no_grad():
                    out = _run_provider(key, mod, A, B)
                diff = (out.float() - ref.float()).abs()
                max_abs = diff.max().item()
                ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
            except Exception as e:
                ok, max_abs = False, float("nan")
                cells.append(f"{name}=[ERROR:{type(e).__name__}: {str(e)[:50]}]")
                all_pass = False
                continue
            cells.append(f"{name}=[{'PASS' if ok else 'FAIL'} {max_abs:.2e}]")
            if not ok:
                all_pass = False
        print(f"[TEST] {label:<14} " + "  ".join(cells))

    # Broadcasting paths (optimized + baselines vs torch ref)
    bcast_cases = []
    # 2D x 2D
    bcast_cases.append(("2Dx2D",
        torch.randn(128, 64, device="npu", dtype=torch.float16),
        torch.randn(64, 256, device="npu", dtype=torch.float16)))
    # 2D x 3D
    bcast_cases.append(("2Dx3D",
        torch.randn(128, 64, device="npu", dtype=torch.float16),
        torch.randn(4, 64, 256, device="npu", dtype=torch.float16)))
    # 3D x 2D
    bcast_cases.append(("3Dx2D",
        torch.randn(4, 128, 64, device="npu", dtype=torch.float16),
        torch.randn(64, 256, device="npu", dtype=torch.float16)))

    for label, A, B in bcast_cases:
        with torch.no_grad():
            ref = _torch_ref(A, B)
        cells = []
        for key, name, mod in variants:
            try:
                with torch.no_grad():
                    out = _run_provider(key, mod, A, B)
                ok = torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
            except Exception as e:
                ok = False
                cells.append(f"{name}=[ERROR:{type(e).__name__}: {str(e)[:50]}]")
                all_pass = False
                continue
            cells.append(f"{name}=[{'PASS' if ok else 'FAIL'}]")
            if not ok:
                all_pass = False
        print(f"[TEST] {label:<14} " + "  ".join(cells))

    print("[TEST] All tests PASSED" if all_pass else "[TEST] Some tests FAILED")
    return all_pass


# --- Performance benchmark ---------------------------------------------------

def bench():
    """@perf_report benchmark. Table format matches generate_report.py:
    single 'label' shape column + recognized method names. Each provider is
    isolated — a failing provider yields inf for that cell, not a crash."""
    from triton.testing import perf_report, Benchmark, do_bench

    if not torch.npu.is_available():
        print("[BENCH] NPU not available — skipping benchmark")
        return

    # (label, batch, M, N, K)  — labels must contain NO spaces (parser splits on whitespace)
    bench_shapes = [
        ("b2-m128-n128-k64",     2, 128, 128, 64),
        ("b4-m256-n256-k128",    4, 256, 256, 128),
        ("b8-m512-n512-k256",    8, 512, 512, 256),
        ("b16-m512-n1024-k512", 16, 512, 1024, 512),
        ("b32-m512-n2048-k1024",32, 512, 2048, 1024),
    ]

    @perf_report(
        Benchmark(
            x_names=["label"],
            x_vals=[s[0] for s in bench_shapes],
            line_arg="provider",
            line_vals=[k for k, _, _ in _PROVIDERS],
            line_names=[name for _, name, _ in _PROVIDERS],
            styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
            ylabel="Latency (ms)",
            plot_name="bmm_benchmark",
            args={},
        )
    )
    def bmm_benchmark(label, provider):
        _, batch, M, N, K = next(s for s in bench_shapes if s[0] == label)
        A = torch.randn(batch, M, K, device="npu", dtype=torch.float16)
        B = torch.randn(batch, K, N, device="npu", dtype=torch.float16)
        mod = dict((k, m) for k, _, m in _PROVIDERS)[provider]
        try:
            return do_bench(lambda: _run_provider(provider, mod, A, B),
                            quantiles=[0.5, 0.2, 0.8])[0]
        except Exception as e:
            # One provider failing on one shape must not abort the table.
            print(f"[BENCH] {provider} failed on {label}: {type(e).__name__}: {e}")
            return float("inf")

    bmm_benchmark.run(save_path=str(_DIR), show_plots=False, print_data=True)


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark and test BMM kernels")
    parser.add_argument("--test", action="store_true", help="Correctness only")
    parser.add_argument("--bench", action="store_true", help="Benchmark only")
    args = parser.parse_args()

    run_test = args.test or not args.bench
    run_bench = args.bench or not args.test

    if run_test:
        print("=" * 60)
        print("Correctness Test")
        print("=" * 60)
        test_correctness()

    if run_bench:
        print()
        print("=" * 60)
        print("Performance Benchmark")
        print("=" * 60)
        bench()
