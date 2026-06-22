"""
profile_kernels.py -- l1_19 ReLU

Compares three implementations on Ascend NPU hardware:
  torch_ref  : torch.nn.functional.relu (ACL built-in path)
  baseline   : original Triton kernel (19_ReLU.py, autotune + 1D grid)
  optimized  : persistent-grid + fp32-upcast kernel (opt_19_ReLU.py)

Usage:
    python profile_kernels.py            # unit test + benchmark + saved figure
    python profile_kernels.py --test     # correctness only
    python profile_kernels.py --bench    # benchmark only
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

_DIR = Path(__file__).parent


def _load(fname):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_baseline_mod1 = _load(_DIR / "23_Softmax.py")
_baseline_mod2 = _load(_DIR / "base_23_Softmax.py")
_optimized_mod = _load(_DIR / "opt_23_Softmax.py")

# ── runner functions ────────────────────────────────────────────────────────────


def _run_torch_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch built-in ReLU -- routes to Huawei ACL path on Ascend."""
    return torch.nn.functional.softmax(x, dim=1)


_baseline_model1 = _baseline_mod1.ModelNew()


def _run_baseline1(x: torch.Tensor) -> torch.Tensor:

    return _baseline_model1(x)


_baseline_model2 = _baseline_mod2.ModelNew()


def _run_baseline2(x: torch.Tensor) -> torch.Tensor:

    return _baseline_model2(x)


# Instantiate once — avoids __init__ overhead on every benchmark call.
_optimized_model = _optimized_mod.ModelNew()


def _run_optimized(x: torch.Tensor) -> torch.Tensor:
    """Optimized Triton kernel: two-path dispatch (direct / persistent) +
    fp32 upcast for RVECEX path + bucketed autotune key."""
    return _optimized_model(x)


# ── benchmark shapes ────────────────────────────────────────────────────────────

_BENCH_SHAPES = [
    # (label,             n_elements, n_cols)
    ("N=256*256", 256, 256),
    ("N=512*512", 512, 512),
    ("N=1024*1024", 1024, 1024),
    ("N=60000*2048", 60000, 2048),
    ("N=60000*2048", 60000, 5096),
]

# ── perf_report benchmark ───────────────────────────────────────────────────────


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="relu_perf",
        args={},
    ))
def benchmark(label, mode):
    _, n_samples, n_cols = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = torch.rand(n_samples, n_cols, device="npu", dtype=torch.float32)

    if mode == "torch_ref":

        def fn():
            return _run_torch_ref(x)
    elif mode == "baseline1":

        def fn():
            return _run_baseline1(x)
    elif mode == "baseline2":

        def fn():
            return _run_baseline2(x)
    else:

        def fn():
            return _run_optimized(x)

    # do_bench returns seconds; perf_report handles ylabel labelling
    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")


# ── unit test ───────────────────────────────────────────────────────────────────


def unit_test():
    """Verify baseline and optimized match torch.nn.functional.softmax across shapes."""
    torch.manual_seed(42)
    any_fail = False

    print("=== Unit Test: l1_23_Softmax ===")
    for label, n_samples, n_cols in _BENCH_SHAPES:
        x = torch.rand(n_samples, n_cols, device="npu", dtype=torch.float32)
        ref = _run_torch_ref(x.clone())
        base1 = _run_baseline1(x.clone())
        base2 = _run_baseline2(x.clone())
        opt = _run_optimized(x.clone())

        ok_b1 = torch.allclose(ref, base1, atol=1e-2, rtol=1e-2)
        ok_b2 = torch.allclose(ref, base2, atol=1e-2, rtol=1e-2)
        ok_o = torch.allclose(ref, opt, atol=1e-2, rtol=1e-2)
        max_b1 = (ref - base1).abs().max().item()
        max_b2 = (ref - base2).abs().max().item()
        max_o = (ref - opt).abs().max().item()

        print(
            f"  {label:<28}  baseline1 [{'PASS' if ok_b1 else 'FAIL'}]  "
            f"baseline2 [{'PASS' if ok_b2 else 'FAIL'}]  "
            f"optimized [{'PASS' if ok_o else 'FAIL'}]  "
            f"maxDelta_base1={max_b1:.2e} maxDelta_base2={max_b2:.2e}  maxDelta_opt={max_o:.2e}"
        )

        if not ok_b1 or not ok_b2 or not ok_o:
            any_fail = True

    if any_fail:
        print("FAILED: some shapes did not pass correctness check")
        sys.exit(1)
    print("All shapes PASSED")


# ── entry point ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Profile l1_19_ReLU kernels")
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
