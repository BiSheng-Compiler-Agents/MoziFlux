import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "40_Matmul_Scaling_ResidualAdd.py"
BASE_FILE = ROOT / "base_40_Matmul_Scaling_ResidualAdd.py"
OPT_FILE = ROOT / "opt_40_Matmul_Scaling_ResidualAdd.py"

_BENCH_SHAPES = [
    ("small_acl_128x512x512", 128, 512, 512, 0.5),
    ("irregular_triton_257x512x512", 257, 512, 512, 0.5),
    ("required_acl_16384x4096x4096", 16384, 4096, 4096, 0.5),
]

_PROVIDERS = [
    "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
]
_MODEL_CACHE = {}
_MODULE_CACHE = {}


def _load(path: Path, key: str):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"k_l2_40_{key}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def _safe_load(path: Path, key: str):
    try:
        return _load(path, key)
    except Exception as exc:
        print(f"INFO provider_unavailable {key}: {type(exc).__name__}")
        return None


def _model(key: str, in_features: int, out_features: int,
           scaling_factor: float):
    cache_key = (key, in_features, out_features, scaling_factor)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    path = {"input": INPUT_FILE, "base": BASE_FILE, "opt": OPT_FILE}[key]
    mod = _safe_load(path, key)
    if mod is None:
        _MODEL_CACHE[cache_key] = None
        return None
    torch.manual_seed(0)
    model = mod.ModelNew(in_features, out_features,
                         scaling_factor).eval().npu()
    _MODEL_CACHE[cache_key] = model
    return model


def _make_inputs(M: int, K: int):
    torch.manual_seed(123)
    return torch.rand(M, K, device="npu", dtype=torch.float32)


def _ref_model(in_features: int, out_features: int, scaling_factor: float):
    return _model("input", in_features, out_features, scaling_factor)


def _run_torch_ref(x, in_features: int, out_features: int,
                   scaling_factor: float):
    model = _ref_model(in_features, out_features, scaling_factor)
    return torch.nn.functional.linear(
        x, model.matmul.weight,
        model.matmul.bias) * (1.0 + float(scaling_factor))


def _run_provider(provider: str, x, in_features: int, out_features: int,
                  scaling_factor: float):
    if provider == "PyTorch / ACL":
        return _run_torch_ref(x, in_features, out_features, scaling_factor)
    key = {
        "Baseline Triton1": "input",
        "Baseline Triton2": "base",
        "Optimized Triton": "opt",
    }[provider]
    model = _model(key, in_features, out_features, scaling_factor)
    if model is None:
        raise RuntimeError("provider unavailable")
    return model(x)


def _shape_by_label(label: str):
    for item in _BENCH_SHAPES:
        if item[0] == label:
            return item
    raise KeyError(label)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok = True
    for label, M, K, N, scale in _BENCH_SHAPES:
        x = _make_inputs(M, K)
        ref = _run_torch_ref(x, K, N, scale)
        torch.npu.synchronize()
        for provider in _PROVIDERS[1:]:
            try:
                y = _run_provider(provider, x, K, N, scale)
                torch.npu.synchronize()
                diff = _max_abs(y, ref)
                passed = math.isfinite(diff) and diff <= 1e-2
                print(
                    f"UNIT {provider} {label} max_abs_diff={diff:.6g} {'PASS' if passed else 'MISMATCH'}"
                )
                if provider == "Optimized Triton" and not passed:
                    ok = False
            except Exception as exc:
                print(
                    f"INFO unit_provider_skip {provider} {label}: {type(exc).__name__}"
                )
                if provider == "Optimized Triton":
                    ok = False
        del x, ref
        torch.npu.empty_cache()
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_ms(fn, warmup=5, rep=20):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(rep):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - start) * 1000.0 / rep


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=_PROVIDERS,
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="l2_40_matmul_scaling_residualadd",
        args={},
    ))
def benchmark(label, provider):
    _, M, K, N, scale = _shape_by_label(label)
    x = _make_inputs(M, K)
    if provider != "PyTorch / ACL":
        try:
            y = _run_provider(provider, x, K, N, scale)
            torch.npu.synchronize()
            ref = _run_torch_ref(x, K, N, scale)
            diff = _max_abs(y, ref)
            if provider == "Optimized Triton" and diff > 1e-2:
                print(
                    f"INFO benchmark_skip_mismatch {provider} {label} diff={diff:.6g}"
                )
                return float("inf")
        except Exception as exc:
            print(
                f"INFO benchmark_provider_unavailable {provider} {label}: {type(exc).__name__}"
            )
            return float("inf")

    def fn():
        return _run_provider(provider, x, K, N, scale)

    try:
        return _time_ms(fn)
    except Exception as exc:
        print(
            f"INFO benchmark_provider_failed {provider} {label}: {type(exc).__name__}"
        )
        return float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not args.test and not args.bench:
        args.test = args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        benchmark.run(print_data=True, show_plots=False, save_path=str(ROOT))


if __name__ == "__main__":
    main()
