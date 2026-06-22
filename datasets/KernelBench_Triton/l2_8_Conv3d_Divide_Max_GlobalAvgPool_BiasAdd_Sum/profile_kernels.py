"""
profile_kernels.py -- benchmark and unit test for
l2_8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

Compares four implementations:
  torch_ref  : pure PyTorch (Conv3d + divide + MaxPool + GlobalAvgPool + bias + sum)
  baseline1  : 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py
                 (Triton kernel, C <= 256 power-of-2 fast path)
  baseline2  : base_8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py
                 (Triton kernel, generic path with C==16 GROUP_TILES)
  optimized  : opt_8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py
                 (optimized Triton kernel with persistent grid + bias_sum)

Usage:
    python profile_kernels.py           # unit test + benchmark
    python profile_kernels.py --test    # correctness check only
    python profile_kernels.py --bench   # benchmark only (skips unit test)
"""
import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.testing

# ── Load baseline and optimized modules from sibling files ─────────────────────
_DIR = Path(__file__).parent


def _load(fname):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_baseline1 = _load(_DIR / "8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py")
_baseline2 = _load(_DIR /
                   "base_8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py")
_optimized = _load(_DIR /
                   "opt_8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py")

# ── Shapes to benchmark / unit test ────────────────────────────────────────────
# Fixed shape: in_channels=8, out_channels=16, D=16, H=64, W=64
# B is swept to measure scaling.
_BENCH_SHAPES = [
    # label              B
    ("B=1", 1),
    ("B=4", 4),
    ("B=16", 16),
    ("B=32", 32),
    ("B=64", 64),
    ("B=128", 128),
    ("B=256", 256),
]

# Canonical shapes (from module-level constants)
_IN_CHANNELS = _baseline1.in_channels  # 8
_OUT_CHANNELS = _baseline1.out_channels  # 16
_DEPTH = _baseline1.depth  # 16
_HEIGHT = _baseline1.height  # 64
_WIDTH = _baseline1.width  # 64
_KERNEL_SIZE = _baseline1.kernel_size  # (3,3,3)
_DIVISOR = _baseline1.divisor  # 2.0
_POOL_SIZE = _baseline1.pool_size  # (2,2,2)
_BIAS_SHAPE = _baseline1.bias_shape  # (16,1,1,1)
_SUM_DIM = _baseline1.sum_dim  # 1
# ── PyTorch reference ──────────────────────────────────────────────────────────
# Uses the baselines' own Conv3d weights + bias to guarantee weight identity.
# Only the post-conv ops (MaxPool, GlobalAvgPool, bias-add, sum) are pure PyTorch.


def _run_torch_ref(x: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference: reuses baseline1's Conv3d weights/bias,
    then does MaxPool + GlobalAvgPool + bias-add + sum in pure PyTorch.
    This guarantees identical conv output before the post-conv ops."""
    _ensure_models_on_npu()
    with torch.no_grad():
        # Conv3d with folded division (same as baseline1)
        m = _baseline1_model
        w = m.conv.weight
        b = m.conv.bias
        x = F.conv3d(x,
                     w / _DIVISOR,
                     None if b is None else b / _DIVISOR,
                     stride=m.conv.stride,
                     padding=m.conv.padding,
                     dilation=m.conv.dilation,
                     groups=m.conv.groups)
        x = m.max_pool(x)
        x = m.global_avg_pool(x)  # [B, C, 1, 1, 1]
        x = (x + m.bias).reshape(x.shape[0], x.shape[1])  # [B, C]
        x = x.sum(dim=1, keepdim=True)  # [B, 1]
        return x.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]


# ── Triton kernel interfaces ───────────────────────────────────────────────────
_baseline1_model = None
_baseline2_model = None
_optimized_model = None


def _ensure_models_on_npu():
    """Lazily move models to NPU on first call (avoids CPU/NPU mismatch)."""
    global _baseline1_model, _baseline2_model, _optimized_model
    if _baseline1_model is None:
        torch.manual_seed(0)
        _baseline1_model = _baseline1.ModelNew(
            *_baseline1.get_init_inputs()).to(device="npu",
                                              dtype=torch.float32).eval()
        torch.manual_seed(0)
        _baseline2_model = _baseline2.ModelNew(
            *_baseline2.get_init_inputs()).to(device="npu",
                                              dtype=torch.float32).eval()
        torch.manual_seed(0)
        _optimized_model = _optimized.ModelNew(
            *_optimized.get_init_inputs()).to(device="npu",
                                              dtype=torch.float32).eval()


def _run_baseline1(x):
    _ensure_models_on_npu()
    return _baseline1_model(x)


def _run_baseline2(x):
    _ensure_models_on_npu()
    return _baseline2_model(x)


def _run_optimized(x):
    _ensure_models_on_npu()
    return _optimized_model(x)


# ── perf_report benchmark ─────────────────────────────────────────────────────
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
        plot_name="conv3d_divide_max_gap_biasadd_sum_perf",
        args={},
    ))
def benchmark(label, mode):
    _, B = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = torch.rand(B,
                   _IN_CHANNELS,
                   _DEPTH,
                   _HEIGHT,
                   _WIDTH,
                   device="npu",
                   dtype=torch.float32)

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

    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")


# ── Unit test ──────────────────────────────────────────────────────────────────
def unit_test():
    print("=" * 60)
    print("UNIT TEST — correctness vs torch_ref (pure PyTorch)")
    print("=" * 60)
    torch.manual_seed(42)
    any_fail = False

    # Pre-warm torch_ref cache (seed=0 matches model weight init)
    torch.manual_seed(0)
    _run_torch_ref(
        torch.rand(1,
                   _IN_CHANNELS,
                   _DEPTH,
                   _HEIGHT,
                   _WIDTH,
                   device="npu",
                   dtype=torch.float32))

    for label, B in _BENCH_SHAPES:
        torch.manual_seed(42)
        x = torch.rand(B,
                       _IN_CHANNELS,
                       _DEPTH,
                       _HEIGHT,
                       _WIDTH,
                       device="npu",
                       dtype=torch.float32)
        with torch.no_grad():
            ref = _run_torch_ref(x)
            base1 = _run_baseline1(x.clone())
            base2 = _run_baseline2(x.clone())
            opt = _run_optimized(x.clone())

        ok_base1 = torch.allclose(ref, base1, atol=1e-3, rtol=1e-3)
        ok_base2 = torch.allclose(ref, base2, atol=1e-3, rtol=1e-3)
        ok_opt = torch.allclose(ref, opt, atol=1e-3, rtol=1e-3)
        max_base1 = (ref - base1).abs().max().item()
        max_base2 = (ref - base2).abs().max().item()
        max_opt = (ref - opt).abs().max().item()

        sb1 = "PASS" if ok_base1 else "FAIL"
        sb2 = "PASS" if ok_base2 else "FAIL"
        so = "PASS" if ok_opt else "FAIL"
        print(f"  {label:<22}  baseline1 [{sb1}] max\u0394={max_base1:.2e}   "
              f"baseline2 [{sb2}] max\u0394={max_base2:.2e}   "
              f"optimized [{so}] max\u0394={max_opt:.2e}")
        if not ok_base1 or not ok_base2 or not ok_opt:
            any_fail = True

    print()
    if any_fail:
        print("RESULT: FAIL — some kernels produced incorrect output.")
    else:
        print("RESULT: PASS — all kernels match torch_ref.")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Profile Conv3D Divide Max GlobalAvgPool BiasAdd Sum: torch_ref vs baseline vs optimized Triton"
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
        benchmark.run(save_path=str(_DIR), print_data=True)


if __name__ == "__main__":
    main()
