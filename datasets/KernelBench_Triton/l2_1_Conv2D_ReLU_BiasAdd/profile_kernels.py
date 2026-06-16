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


_baseline = _load(_DIR / "1_Conv2D_ReLU_BiasAdd.py")
_optimized = _load(_DIR / "opt_1_Conv2D_ReLU_BiasAdd.py")

# ── Baseline kernel interface ──────────────────────────────────────────────────
_baseline_kernel = _baseline._relu_add_bias_kernel


def _run_baseline(x: torch.Tensor, bias_flat: torch.Tensor) -> torch.Tensor:
    """Run the baseline 2D-grid kernel directly.

    The baseline kernel maps one program per (n,c,h) row, so gridX = N*C*H.
    Ascend FFTS caps any grid dimension at 65535. When N*C*H exceeds that
    (e.g. N=128, C=128, H=126 → 2,064,384) we loop over N and launch one
    sub-grid per batch item, each with gridX = C*H which is always ≤ 65535
    for realistic channel/spatial sizes.
    """
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    block_w = 128 if W >= 128 else (64 if W >= 64 else 32)
    w_grid = triton.cdiv(W, block_w)

    if N * C * H <= 65535:
        _baseline_kernel[(N * C * H, w_grid)](
            x,
            y,
            bias_flat,
            N,
            C,
            H,
            W,
            BLOCK_W=block_w,
            num_warps=4,
        )
    else:
        # Chunk over N: each sub-launch has gridX = C*H
        for n in range(N):
            x_n = x[n:n + 1]  # [1, C, H, W] — still contiguous
            y_n = y[n:n + 1]
            _baseline_kernel[(C * H, w_grid)](
                x_n,
                y_n,
                bias_flat,
                1,
                C,
                H,
                W,
                BLOCK_W=block_w,
                num_warps=4,
            )
    return y


# ── Optimised kernel interface ─────────────────────────────────────────────────
_run_optimized = _optimized._relu_add_bias_triton


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
    ("N128-C128-126x126", 128, 128, 126, 126),  # HW=15876 loop, benchmark
]


# perf_report sweeps over x_vals, one curve per line_val (mode).
# It prints a table and saves a PNG automatically.
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=["torch_ref", "baseline", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
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
        fn = lambda: _run_torch_ref(x, bias)  # noqa: E731
    elif mode == "baseline":
        fn = lambda: _run_baseline(x, bias_flat)  # noqa: E731
    else:
        fn = lambda: _run_optimized(x, bias)  # noqa: E731

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
        base = _run_baseline(x.clone(), bias_flat)
        opt = _run_optimized(x.clone(), bias)

        ok_base = torch.allclose(ref, base, atol=1e-2, rtol=1e-2)
        ok_opt = torch.allclose(ref, opt, atol=1e-2, rtol=1e-2)
        max_base = (ref - base).abs().max().item()
        max_opt = (ref - opt).abs().max().item()

        sb = "PASS" if ok_base else "FAIL"
        so = "PASS" if ok_opt else "FAIL"
        print(f"  {label:<24}  baseline [{sb}] maxΔ={max_base:.2e}   "
              f"optimized [{so}] maxΔ={max_opt:.2e}")
        if not ok_base or not ok_opt:
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
