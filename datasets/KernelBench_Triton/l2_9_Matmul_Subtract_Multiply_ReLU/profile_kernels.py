"""
profile_kernels.py -- Matmul_Subtract_Multiply_ReLU (l2_9)

Compares four implementations on Ascend NPU:
  torch_ref  : pure PyTorch (F.linear + sub + mul + relu) using baseline's own Linear weights
  baseline1  : 9_Matmul_Subtract_Multiply_ReLU.py via ModelNew (Triton, 2D grid)
  baseline2  : base_9_Matmul_Subtract_Multiply_ReLU.py via ModelNew (Triton, larger tiles)
  optimized  : opt_9_Matmul_Subtract_Multiply_ReLU.py via ModelNew (1D grid + swizzle)

Usage:
    python profile_kernels.py           # unit test + benchmark
    python profile_kernels.py --test    # correctness check only
    python profile_kernels.py --bench   # benchmark only (skip unit test)
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


_baseline1 = _load(_DIR / "9_Matmul_Subtract_Multiply_ReLU.py")
_baseline2 = _load(_DIR / "base_9_Matmul_Subtract_Multiply_ReLU.py")
_optimized = _load(_DIR / "opt_9_Matmul_Subtract_Multiply_ReLU.py")

# ── Shapes to benchmark / unit test ────────────────────────────────────────────
# Each entry: (label, M, K, N). Small shapes for fast CI simulation.
_BENCH_SHAPES = [
    ("M128-K256-N256", 128, 256, 256),  # tiny — 1 tile per dim
    ("M256-K512-N512", 256, 512, 512),  # small
    ("M512-K1024-N1024", 512, 1024, 1024),  # medium — benchmark shape
]

# Canonical hyper-parameters
_SUB_VAL = _baseline1.subtract_value  # 2.0
_MUL_VAL = _baseline1.multiply_value  # 1.5

# ── Per-shape model cache ──────────────────────────────────────────────────────
# Keyed by (K, N) — each shape needs its own Linear with matching in/out features.
_model_cache: dict[tuple[int, int], tuple] = {}


def _ensure_models_on_npu(K: int, N: int):
    """Lazily create models for shape (K, N) on NPU."""
    if (K, N) in _model_cache:
        return
    torch.manual_seed(0)
    m1 = _baseline1.ModelNew(K, N, _SUB_VAL,
                             _MUL_VAL).to(device="npu",
                                          dtype=torch.float16).eval()
    torch.manual_seed(0)
    m2 = _baseline2.ModelNew(K, N, _SUB_VAL,
                             _MUL_VAL).to(device="npu",
                                          dtype=torch.float16).eval()
    torch.manual_seed(0)
    m3 = _optimized.ModelNew(K, N, _SUB_VAL,
                             _MUL_VAL).to(device="npu",
                                          dtype=torch.float16).eval()
    _model_cache[(K, N)] = (m1, m2, m3)


def _run_torch_ref(x: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Pure PyTorch/ACL reference: linear + sub + mul + relu.
    Reuses baseline1's Linear weight/bias to guarantee identical matmul input."""
    _ensure_models_on_npu(K, N)
    m1, _, _ = _model_cache[(K, N)]
    with torch.no_grad():
        out = F.linear(x, m1.linear.weight, m1.linear.bias)
        return torch.relu((out - _SUB_VAL) * _MUL_VAL)


def _run_baseline1(x: torch.Tensor, K: int, N: int):
    _ensure_models_on_npu(K, N)
    m1, _, _ = _model_cache[(K, N)]
    return m1(x)


def _run_baseline2(x: torch.Tensor, K: int, N: int):
    _ensure_models_on_npu(K, N)
    _, m2, _ = _model_cache[(K, N)]
    return m2(x)


def _run_optimized(x: torch.Tensor, K: int, N: int):
    _ensure_models_on_npu(K, N)
    _, _, m3 = _model_cache[(K, N)]
    return m3(x)


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
        plot_name="l2_9_Matmul_Sub_Mul_ReLU_perf",
        args={},
    ))
def benchmark(label, mode):
    _, M, K, N = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = torch.rand(M, K, device="npu", dtype=torch.float16)

    if mode == "torch_ref":
        fn = lambda: _run_torch_ref(x, K, N)  # noqa: E731
    elif mode == "baseline1":
        fn = lambda: _run_baseline1(x.clone(), K, N)  # noqa: E731
    elif mode == "baseline2":
        fn = lambda: _run_baseline2(x.clone(), K, N)  # noqa: E731
    else:
        fn = lambda: _run_optimized(x.clone(), K, N)  # noqa: E731

    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")


# ── Unit test ──────────────────────────────────────────────────────────────────
def unit_test():
    print("=" * 60)
    print("UNIT TEST — correctness vs torch_ref (pure PyTorch)")
    print("=" * 60)
    torch.manual_seed(42)
    any_fail = False

    for label, M, K, N in _BENCH_SHAPES:
        # Pre-warm model cache for this shape
        torch.manual_seed(0)
        _ensure_models_on_npu(K, N)

        torch.manual_seed(42)
        x = torch.rand(M, K, device="npu", dtype=torch.float16) * 4 - 2
        with torch.no_grad():
            ref = _run_torch_ref(x, K, N)
            base1 = _run_baseline1(x.clone(), K, N)
            base2 = _run_baseline2(x.clone(), K, N)
            opt = _run_optimized(x.clone(), K, N)

        ok_base1 = torch.allclose(ref, base1, atol=1e-2, rtol=1e-2)
        ok_base2 = torch.allclose(ref, base2, atol=1e-2, rtol=1e-2)
        ok_opt = torch.allclose(ref, opt, atol=1e-2, rtol=1e-2)
        max_base1 = (ref - base1).abs().max().item()
        max_base2 = (ref - base2).abs().max().item()
        max_opt = (ref - opt).abs().max().item()

        sb1 = "PASS" if ok_base1 else "FAIL"
        sb2 = "PASS" if ok_base2 else "FAIL"
        so = "PASS" if ok_opt else "FAIL"
        print(f"  {label:<28}  baseline1 [{sb1}] max\u0394={max_base1:.2e}  "
              f"baseline2 [{sb2}] max\u0394={max_base2:.2e}  "
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
        "Profile Matmul Sub Mul Relu: torch_ref vs baseline vs optimized Triton"
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
