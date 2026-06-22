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
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        return m
    except Exception as e:
        if optional:
            print(
                f"INFO optional_provider {fname} unavailable {type(e).__name__}"
            )
            return None
        raise


_baseline_mod1 = _load("18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py")
_baseline_mod2 = _load("base_18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py",
                       optional=True)
_optimized_mod = _load("opt_18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py")

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
_MODELS = {}
_REF_MODELS = {}

_BENCH_SHAPES = [
    ("small_B64_I512_O512", 64, 512, 512),
    ("medium_B256_I2048_O2048", 256, 2048, 2048),
    ("target_B1024_I8192_O8192", 1024, 8192, 8192),
]


def _init_args(I, O):  # noqa: E741
    return [I, O]


def _model(key, I, O):  # noqa: E741
    ck = (key, I, O)
    if ck not in _MODELS:
        mod = _MODS.get(key)
        if mod is None:
            return None
        torch.manual_seed(0)
        _MODELS[ck] = mod.ModelNew(*_init_args(I, O)).to(
            device="npu", dtype=torch.float32).eval()
    return _MODELS[ck]


def _ref_model(I, O):  # noqa: E741
    ck = (I, O)
    if ck not in _REF_MODELS:
        torch.manual_seed(0)
        layer = torch.nn.Linear(I, O).to(device="npu",
                                         dtype=torch.float32).eval()
        _REF_MODELS[ck] = layer
    return _REF_MODELS[ck]


def _make_inputs(B, I):  # noqa: E741
    torch.manual_seed(42)
    return torch.rand(B, I, device="npu", dtype=torch.float32)


def _run_torch_ref(x, I, O):  # noqa: E741
    layer = _ref_model(I, O)
    return F.linear(x, layer.weight, layer.bias).sum(dim=1, keepdim=True)


def _run_provider(key, x, I, O):  # noqa: E741
    if key == "torch_ref":
        return _run_torch_ref(x, I, O)
    m = _model(key, I, O)
    if m is None:
        raise RuntimeError("provider_unavailable")
    return m(x)


def _check_close(ref, out, key, label):
    diff = (ref.float() - out.float()).abs()
    max_abs = diff.max().item()
    denom = ref.float().abs().clamp_min(1.0)
    max_rel = (diff / denom).max().item()
    ok = torch.allclose(ref.float(), out.float(), atol=1e-2, rtol=1e-4)
    if key == "optimized":
        word = "PASS" if ok else "MISMATCH"
    else:
        word = "PASS" if ok else "SKIP value_delta"
    print(
        f"TEST {key} {label} {word} max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
    )
    return ok


def unit_test():
    print("=== Unit Test: l2_18 matmul-sum singleton reductions ===")
    optimized_ok = True
    for label, B, I, O in _BENCH_SHAPES:  # noqa: E741
        x = _make_inputs(B, I)
        with torch.no_grad():
            ref = _run_torch_ref(x, I, O)
        for key, _name in _VARIANTS:
            try:
                with torch.no_grad():
                    out = _run_provider(key, x, I, O)
                ok = _check_close(ref, out, key, label)
                if key == "optimized" and not ok:
                    optimized_ok = False
            except Exception as e:
                if key == "optimized":
                    optimized_ok = False
                    print(
                        f"TEST {key} {label} ERROR {type(e).__name__} {str(e)[:80]}"
                    )
                else:
                    print(
                        f"TEST {key} {label} SKIP provider_unavailable {type(e).__name__}"
                    )
    print("UNIT_TEST PASS" if optimized_ok else "UNIT_TEST_FAILED")
    return optimized_ok


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=[k for k, _ in _PROVIDERS],
        line_names=[n for _, n in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "-")],
        ylabel="Latency (ms)",
        plot_name="l2_18_matmul_sum_perf",
        args={},
    ))
def benchmark(label, mode):
    _, B, I, O = next(s for s in _BENCH_SHAPES if s[0] == label)  # noqa: E741
    x = _make_inputs(B, I)
    try:
        return triton.testing.do_bench(lambda: _run_provider(mode, x, I, O),
                                       warmup=5,
                                       rep=20,
                                       return_mode="mean")
    except Exception as e:
        print(f"INFO bench {mode} {label} inf {type(e).__name__}")
        return float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    run_test = args.test or not args.bench
    run_bench = args.bench or not args.test
    if run_test:
        unit_test()
    if run_bench:
        benchmark.run(save_path=str(_DIR), print_data=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
