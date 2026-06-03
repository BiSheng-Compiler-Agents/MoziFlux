#!/usr/bin/env python3
"""
profile_kernels.py — HingeLoss benchmark & correctness test for Ascend NPU.

Correctness checks EVERY variant (baseline1, baseline2, optimized) against the
torch reference. The benchmark is resilient: a provider that fails on a shape
yields inf for that cell instead of aborting the whole run.

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

try:
    import triton  # noqa: F401
    from triton.testing import do_bench, perf_report, Benchmark
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False
    do_bench = perf_report = Benchmark = None


def _load(fname):
    """Load a sibling kernel file under a safe module identifier."""
    path = _DIR / fname
    spec = importlib.util.spec_from_file_location("k_" + path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# baseline1 == input kernel, baseline2 == golden reference (base_*), optimized == opt_*
_baseline1 = _load("100_HingeLoss.py")
_baseline2 = _load("base_100_HingeLoss.py")
_optimized = _load("opt_100_HingeLoss.py")

_PROVIDERS = [
    ("torch_ref", "PyTorch / ACL"),
    ("baseline1", "Baseline Triton1"),
    ("baseline2", "Baseline Triton2"),
    ("optimized", "Optimized Triton"),
]
_VARIANTS = [(k, n) for k, n in _PROVIDERS if k != "torch_ref"]
_MODS = {"baseline1": _baseline1, "baseline2": _baseline2, "optimized": _optimized}


def hinge_loss_torch(pred, targ):
    """Reference hinge loss: mean(max(0, 1 - p*t))."""
    return torch.clamp(1.0 - pred * targ, min=0.0).mean()


def _run_provider(key, pred, targ):
    if key == "torch_ref":
        return hinge_loss_torch(pred, targ)
    return _MODS[key].ModelNew()(pred, targ)


SHAPES = [
    # (label, N) — labels must contain NO spaces (parser splits on whitespace)
    ("tiny",   512),      # direct single-tile (N <= BLOCK)
    ("small",  4096),     # multi-tile, power-of-2
    ("medium", 32768),    # reference shape
    ("large",  131072),   # stress test
    ("nopow2", 5000),     # non-power-of-2 edge case
    ("bench",  32768),    # benchmark shape
]


# ── Correctness: every variant vs torch reference ───────────────
def test_correctness():
    print("=" * 60)
    print("Correctness Test: every variant vs torch reference")
    print("=" * 60)

    if not torch.npu.is_available():
        print("  NPU not available — skipping hardware test")
        return True

    all_pass = True
    for label, N in SHAPES:
        torch.manual_seed(42)
        pred = (torch.rand(N, device="npu", dtype=torch.float32) * 2 - 1)
        targ = (torch.randint(0, 2, (N,), device="npu").float() * 2 - 1)
        ref = hinge_loss_torch(pred, targ).item()

        cells = []
        for key, name in _VARIANTS:
            try:
                with torch.no_grad():
                    out = _run_provider(key, pred, targ).cpu().item()
                diff = abs(out - ref)
                ok = diff < 1e-3 + 1e-3 * abs(ref)
            except Exception as e:
                ok = False
                cells.append(f"{name}=[ERROR:{type(e).__name__}: {str(e)[:50]}]")
                all_pass = False
                continue
            cells.append(f"{name}=[{'PASS' if ok else 'FAIL'} {diff:.2e}]")
            if not ok:
                all_pass = False
        print(f"  [{label:<8} N={N:<7}] " + "  ".join(cells))

    print("─" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()
    return all_pass


# ── Benchmark ───────────────────────────────────────────────────
# Canonical perf_report format: single 'label' shape column + recognized
# method names. Parsed by generate_report.py. Resilient per-provider.
@perf_report(
    Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in SHAPES],
        line_arg="provider",
        line_vals=[k for k, _ in _PROVIDERS],
        line_names=[n for _, n in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="hinge_loss_benchmark",
        args={},
    )
)
def benchmark(label, provider):
    _, N = next(s for s in SHAPES if s[0] == label)
    torch.manual_seed(42)
    pred = (torch.rand(N, device="npu", dtype=torch.float32) * 2 - 1)
    targ = (torch.randint(0, 2, (N,), device="npu").float() * 2 - 1)
    try:
        return do_bench(lambda: _run_provider(provider, pred, targ),
                        quantiles=[0.5, 0.2, 0.8])[0]
    except Exception as e:
        print(f"[BENCH] {provider} failed on {label}: {type(e).__name__}: {e}")
        return float("inf")


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HingeLoss profile")
    parser.add_argument("--test", action="store_true", help="Correctness only")
    parser.add_argument("--bench", action="store_true", help="Benchmark only")
    args = parser.parse_args()

    run_test = args.test or (not args.bench)
    run_bench = args.bench or (not args.test)

    if run_test:
        ok = test_correctness()
        if args.test:
            sys.exit(0 if ok else 1)

    if run_bench:
        print("=" * 60)
        print("Benchmark: HingeLoss Latency")
        print("=" * 60)
        benchmark.run(save_path=str(_DIR), print_data=True)


if __name__ == "__main__":
    main()
