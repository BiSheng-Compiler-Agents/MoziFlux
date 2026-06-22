"""
profile.py — Conv2D → ReLU → BiasAdd  (l2_1_Conv2D_ReLU_BiasAdd)

Compares three implementations of the post-conv activation:

  torch_ref  : nn.functional.relu(x) + bias  (PyTorch/ACL built-in ops)
  baseline   : _relu_add_bias_kernel from 1_Conv2D_ReLU_BiasAdd.py
                 2D grid (N*C*H / BLOCK_ROW, W / BLOCK_W), scalar DIV/REM
                 for channel computation, num_stages=3
  optimized  : _relu_bias_persistent / _relu_bias_loop from opt_1_Conv2D_ReLU_BiasAdd.py
                 Dispatch: HW<=1024 → persistent 32-prog grid-stride;
                           HW>1024  → (N*C,) grid, BLOCK=2048, num_stages=2
                 All decisions validated via cannsim trace_core0.json.

Usage:
    python profile.py           # unit test + perf_report benchmark + saved figure
    python profile.py --test    # correctness check only
    python profile.py --bench   # benchmark only (skips unit test)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

# ── Load baseline and optimized modules from sibling files ─────────────────────
_DIR = Path(__file__).parent


def _load(fname):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_baseline1 = _load(_DIR / "1_Conv2D_ReLU_BiasAdd.py")
_baseline2 = _load(_DIR / "base_1_Conv2D_ReLU_BiasAdd.py")
_optimized = _load(_DIR / "opt_1_Conv2D_ReLU_BiasAdd.py")

# ── Baseline kernel interface ──────────────────────────────────────────────────
_baseline_model1 = _baseline1.ModelNew()


def _run_baseline1(x: torch.Tensor, bias_flat: torch.Tensor) -> torch.Tensor:

    return _baseline_model1(x, bias_flat)


_baseline_model2 = _baseline2.ModelNew()


def _run_baseline2(x: torch.Tensor, bias_flat: torch.Tensor) -> torch.Tensor:

    return _baseline_model2(x, bias_flat)


# ── Optimised kernel interface ─────────────────────────────────────────────────
_optimized_model = _optimized.ModelNew()


def _run_optimized(x, bias):

    return _optimized_model(x, bias)


# ── PyTorch reference ──────────────────────────────────────────────────────────
def _run_torch_ref(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch: relu then add broadcast bias — routes through ACL."""
    return torch.nn.functional.relu(x) + bias  # bias shape [C,1,1] broadcasts


# ── Shapes to benchmark ────────────────────────────────────────────────────────
# x_vals are (N, C, H_out, W_out) tuples — the post-conv tensor shape.
# Covers both dispatch paths:
#   HW <= 1024  → persistent kernel (32-prog grid-stride)
#   HW >  1024  → loop kernel       ((N*C,) grid, BLOCK=2048)
_BENCH_SHAPES = [
    # label              N    C   H_out  W_out
    ("N1-C256-14x14", 1, 256, 14, 14),  # HW=196   persistent
    ("N1-C128-28x28", 1, 128, 28, 28),  # HW=784   persistent
    ("N8-C64-14x14", 8, 64, 14, 14),  # HW=196×8 persistent
    ("N1-C64-56x56", 1, 64, 56, 56),  # HW=3136  loop
    ("N1-C96-56x56", 1, 96, 56, 56),  # HW=3136  loop, non-pow2 C
    ("N1-C32-224x224", 1, 32, 224, 224),  # HW=50176 loop
]


# perf_report sweeps over x_vals, one curve per line_val (mode).
# It prints a table and saves a PNG automatically.
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
        plot_name="relu_bias_add_perf",
        args={},
    ))
def benchmark(label, mode):
    # Look up shape from label
    _, N, C, H_out, W_out = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = torch.rand(N, C, H_out, W_out, device="npu", dtype=torch.float16)
    bias = torch.rand(C, 1, 1, device="npu", dtype=torch.float16)
    bias_flat = bias.reshape(-1)

    if mode == "torch_ref":

        def fn():
            return _run_torch_ref(x, bias)
    elif mode == "baseline1":

        def fn():
            return _run_baseline1(x, bias_flat)
    elif mode == "baseline2":

        def fn():
            return _run_baseline2(x, bias_flat)
    else:

        def fn():
            return _run_optimized(x, bias)

    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")


# ── Unit test ──────────────────────────────────────────────────────────────────
def unit_test():
    print("=" * 60)
    print("UNIT TEST — correctness vs torch.nn.functional.relu + bias")
    print("=" * 60)
    torch.manual_seed(42)
    any_fail = False

    for label, N, C, H_out, W_out in _BENCH_SHAPES:
        x = torch.rand(N, C, H_out, W_out, device="npu",
                       dtype=torch.float16) * 4 - 2
        bias = torch.rand(C, 1, 1, device="npu", dtype=torch.float16) * 0.5
        bias_flat = bias.reshape(-1)

        ref = _run_torch_ref(x, bias)
        base1 = _run_baseline1(x.clone(), bias_flat)
        base2 = _run_baseline2(x.clone(), bias_flat)
        opt = _run_optimized(x.clone(), bias)

        ok_base1 = torch.allclose(ref, base1, atol=1e-2, rtol=1e-2)
        ok_base2 = torch.allclose(ref, base2, atol=1e-2, rtol=1e-2)
        ok_opt = torch.allclose(ref, opt, atol=1e-2, rtol=1e-2)
        max_base1 = (ref - base1).abs().max().item()
        max_base2 = (ref - base2).abs().max().item()
        max_opt = (ref - opt).abs().max().item()

        sb1 = "PASS" if ok_base1 else "FAIL"
        sb2 = "PASS" if ok_base2 else "FAIL"
        so = "PASS" if ok_opt else "FAIL"
        print(f"  {label:<24}  baseline1 [{sb1}] maxΔ={max_base1:.2e}   "
              f"baseline2 [{sb2}] maxΔ={max_base2:.2e}   "
              f"optimized [{so}] maxΔ={max_opt:.2e}")
        if not ok_base1 or not ok_base2 or not ok_opt:
            any_fail = True

    print()
    if any_fail:
        print("RESULT: FAIL — some kernels produced incorrect output.")
        sys.exit(1)
    else:
        print("RESULT: PASS — all kernels match torch reference.")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Profile Conv2D ReLU BiasAdd: torch_ref vs baseline vs optimized Triton"
    )
    parser.add_argument("--test",
                        action="store_true",
                        help="Run correctness unit test only")
    parser.add_argument("--bench",
                        action="store_true",
                        help="Run benchmark only (skip unit test)")
    args = parser.parse_args()

    run_test = args.test or not args.bench
    run_bench = args.bench or not args.test

    if run_test:
        unit_test()
    if run_bench:
        # perf_report prints the table and saves relu_bias_add_perf.png
        benchmark.run(save_path=str(_DIR), print_data=True)


if __name__ == "__main__":
    main()
