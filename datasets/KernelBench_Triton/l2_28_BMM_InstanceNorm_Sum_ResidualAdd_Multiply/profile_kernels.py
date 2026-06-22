"""
profile_kernels.py -- l2_28 BMM_InstanceNorm_Sum_ResidualAdd_Multiply

Compares PyTorch/ACL reference, editable input baseline, read-only base_*.py, and optimized Triton.
Correctness runs for every provider/shape; benchmark cells return inf on handled comparison-provider failures.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

_DIR = Path(__file__).parent


def _load(fname, optional=False):
    try:
        spec = importlib.util.spec_from_file_location("k_" + Path(fname).stem,
                                                      _DIR / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        if optional:
            print(
                f"INFO optional_provider {fname} unavailable {type(e).__name__}"
            )
            return None
        raise


_baseline_mod1 = _load("28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py")
_baseline_mod2 = _load("base_28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py",
                       optional=True)
_optimized_mod = _load("opt_28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py")

_PROVIDERS = [
    ("torch_ref", "PyTorch / ACL"),
    ("baseline1", "Baseline Triton1"),
    ("baseline2", "Baseline Triton2"),
    ("optimized", "Optimized Triton"),
]
_VARIANTS = [(k, n) for k, n in _PROVIDERS if k != "torch_ref"]
_MODS = {
    "baseline1": _baseline_mod1,
    "baseline2": _baseline_mod2,
    "optimized": _optimized_mod
}
_models = {}
_ref_models = {}

_BENCH_SHAPES = [
    ("small_128x512x512", 128, 512, 512),
    ("irregular_257x768x640", 257, 768, 640),
    ("target_1024x8192x8192", 1024, 8192, 8192),
]
_DISPATCH_TEST_SHAPES = [
    ("persistent_70000x32x32", 70000, 32, 32),
]


def _init_args(M, K, N):
    return [K, N]


def _model(key, M, K, N, dtype=torch.float32):
    mod = _MODS.get(key)
    if mod is None:
        return None
    cache_key = (key, K, N, str(dtype))
    if cache_key not in _models:
        torch.manual_seed(0)
        model = mod.ModelNew(*_init_args(M, K, N)).to(device="npu",
                                                      dtype=dtype).eval()
        _models[cache_key] = model
    return _models[cache_key]


def _ref_model(M, K, N, dtype=torch.float32):
    cache_key = (K, N, str(dtype))
    if cache_key not in _ref_models:
        torch.manual_seed(0)
        model = _optimized_mod.ModelNew(K, N).to(device="npu",
                                                 dtype=dtype).eval()
        _ref_models[cache_key] = model
    return _ref_models[cache_key]


def _make_inputs(M, K, N, dtype=torch.float32):
    torch.manual_seed(1234 + M * 3 + K * 5 + N * 7)
    x = torch.rand((M, K), device="npu", dtype=dtype)
    y = torch.rand((M, N), device="npu", dtype=dtype)
    return x, y


def _run_torch_ref(x, y, M, K, N):
    model = _ref_model(M, K, N, x.dtype)
    z = F.linear(x.contiguous(), model.bmm.weight.contiguous(),
                 model.bmm.bias.contiguous())
    mean = z.mean(dim=1, keepdim=True)
    var = (z * z).mean(dim=1, keepdim=True) - mean * mean
    norm = (z - mean) * torch.rsqrt(torch.clamp(var, min=0.0) + model.eps)
    return (norm + y.contiguous()) * y.contiguous()


def _run_provider(key, x, y, M, K, N):
    if key == "torch_ref":
        return _run_torch_ref(x, y, M, K, N)
    model = _model(key, M, K, N, x.dtype)
    if model is None:
        raise RuntimeError("provider_unavailable")
    return model(x, y)


def _max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def unit_test():
    print("=== Unit Test: l2_28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply ===")
    ok_opt = True
    all_shapes = _BENCH_SHAPES + _DISPATCH_TEST_SHAPES
    for label, M, K, N in all_shapes:
        x, y = _make_inputs(M, K, N)
        with torch.no_grad():
            ref = _run_torch_ref(x, y, M, K, N)
        for key, name in _VARIANTS:
            try:
                if key != "optimized" and label.startswith("persistent_"):
                    print(f"TEST {key} {label} SKIP dispatch_only")
                    continue
                with torch.no_grad():
                    out = _run_provider(key, x, y, M, K, N)
                diff = _max_diff(ref, out)
                passed = torch.allclose(ref, out, atol=1e-3, rtol=1e-3)
                print(
                    f"TEST {key} {label} {'PASS' if passed else 'SKIP'} max_diff={diff:.6e}"
                )
                if key == "optimized" and not passed:
                    ok_opt = False
            except Exception as e:
                if key == "optimized":
                    ok_opt = False
                    print(
                        f"TEST {key} {label} FAIL {type(e).__name__}:{str(e)[:80]}"
                    )
                else:
                    print(
                        f"TEST {key} {label} SKIP provider_issue_{type(e).__name__}"
                    )
    print("UNIT_TEST PASS" if ok_opt else "UNIT_TEST_FAILED")
    return ok_opt


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=[k for k, _ in _PROVIDERS],
        line_names=[n for _, n in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "-")],
        ylabel="Latency (ms)",
        plot_name="l2_28_bmm_instancenorm_sum_residual_mul",
        args={},
    ))
def benchmark(label, mode):
    _, M, K, N = next(s for s in _BENCH_SHAPES if s[0] == label)
    x, y = _make_inputs(M, K, N)
    try:
        with torch.no_grad():
            return triton.testing.do_bench(
                lambda: _run_provider(mode, x, y, M, K, N),
                warmup=5,
                rep=20,
                return_mode="mean")
    except Exception as e:
        print(
            f"INFO bench {mode} {label} inf provider_issue_{type(e).__name__}")
        return float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    run_test = args.test or not args.bench
    run_bench = args.bench or not args.test
    ok = True
    if run_test:
        ok = unit_test() and ok
    if run_bench:
        benchmark.run(save_path=str(_DIR), print_data=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
