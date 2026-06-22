"""
profile_kernels.py -- l2_16 ConvTranspose2d + Mish + Add + Hardtanh + Scaling.

Runs parser-compatible correctness and @triton.testing.perf_report benchmarks on Ascend NPU.
"""
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

_DIR = Path(__file__).parent


def _load(fname):
    spec = importlib.util.spec_from_file_location("k_" + Path(fname).stem,
                                                  _DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname):
    try:
        return _load(fname)
    except Exception as exc:
        print(
            f"INFO optional_provider {fname} unavailable {type(exc).__name__}")
        return None


_baseline_mod1 = _load("16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling.py")
_baseline_mod2 = _load_optional(
    "base_16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling.py")
_optimized_mod = _load("opt_16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling.py")

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

# label, B, Cin, Cout, H, W, K, stride, padding, output_padding, add, scale
_BENCH_SHAPES = [
    ("small_direct", 1, 8, 8, 16, 16, 3, 2, 1, 1, 0.5, 2.0),
    ("medium_direct", 4, 32, 32, 64, 64, 3, 2, 1, 1, 0.5, 2.0),
    ("target_persistent", 128, 64, 64, 128, 128, 3, 2, 1, 1, 0.5, 2.0),
]


def _shape_by_label(label):
    for row in _BENCH_SHAPES:
        if row[0] == label:
            return row
    raise KeyError(label)


def _init_args(row):
    _, _B, cin, cout, _H, _W, k, stride, padding, output_padding, add, scale = row
    return [cin, cout, k, stride, padding, output_padding, add, scale]


def _output_numel(row):
    _, B, _cin, cout, H, W, k, stride, padding, output_padding, _add, _scale = row
    hout = (H - 1) * stride - 2 * padding + k + output_padding
    wout = (W - 1) * stride - 2 * padding + k + output_padding
    return B * cout * hout * wout


def _direct_grid_overflows(row, block=4096):
    return math.ceil(_output_numel(row) / block) > 65535


def _make_inputs(row, dtype=torch.float32):
    _, B, cin, _cout, H, W, _k, _stride, _padding, _output_padding, _add, _scale = row
    torch.manual_seed(42)
    x = torch.rand((B, cin, H, W), device="npu", dtype=dtype)
    return (x, )


def _ref_model(row, dtype=torch.float32):
    key = (row[0], dtype)
    if key not in _ref_models:
        _, _B, cin, cout, _H, _W, k, stride, padding, output_padding, _add, _scale = row
        torch.manual_seed(0)
        m = nn.ConvTranspose2d(cin, cout, k, stride,
                               padding, output_padding).to(device="npu",
                                                           dtype=dtype).eval()
        _ref_models[key] = m
    return _ref_models[key]


def _model(provider, row, dtype=torch.float32):
    key = (provider, row[0], dtype)
    if key not in _models:
        mod = _MODS[provider]
        if mod is None:
            raise RuntimeError("provider_unavailable")
        torch.manual_seed(0)
        _models[key] = mod.ModelNew(*_init_args(row)).to(device="npu",
                                                         dtype=dtype).eval()
    return _models[key]


def _run_torch_ref(row, x):
    add = row[-2]
    scale = row[-1]
    y = _ref_model(row, x.dtype)(x)
    y = F.mish(y)
    y = torch.clamp(y + add, -1.0, 1.0)
    return y * scale


def _run_provider(provider, row, x):
    if provider == "torch_ref":
        return _run_torch_ref(row, x)
    if provider in ("baseline1", "baseline2") and _direct_grid_overflows(row):
        raise RuntimeError("grid_guard")
    return _model(provider, row, x.dtype)(x)


def unit_test():
    print("=== Unit Test: l2_16 ConvTranspose2d_Mish_Add_Hardtanh_Scaling ===")
    optimized_ok = True
    for row in _BENCH_SHAPES:
        label = row[0]
        inputs = _make_inputs(row)
        x = inputs[0]
        with torch.no_grad():
            ref = _run_torch_ref(row, x)
        for provider, _name in _VARIANTS:
            if provider in ("baseline1",
                            "baseline2") and _direct_grid_overflows(row):
                print(f"TEST {provider} {label} SKIP grid_guard")
                continue
            if _MODS.get(provider) is None:
                print(f"TEST {provider} {label} SKIP provider_unavailable")
                continue
            try:
                with torch.no_grad():
                    out = _run_provider(provider, row, x)
                diff = (ref.float() - out.float()).abs().max().item()
                ok = torch.allclose(ref.float(),
                                    out.float(),
                                    atol=1e-3,
                                    rtol=1e-3)
                print(
                    f"TEST {provider} {label} {'PASS' if ok else 'FAIL'} max_diff={diff:.6e}"
                )
                if provider == "optimized" and not ok:
                    optimized_ok = False
            except Exception as exc:
                token = "SKIP" if provider != "optimized" else "FAIL"
                print(f"TEST {provider} {label} {token} {type(exc).__name__}")
                if provider == "optimized":
                    optimized_ok = False
        del ref, x, inputs
        torch.npu.empty_cache()
    print("UNIT_TEST PASS" if optimized_ok else "UNIT_TEST_FAILED")
    return optimized_ok


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=[k for k, _ in _PROVIDERS],
        line_names=[n for _, n in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "-")],
        ylabel="Latency (ms)",
        plot_name="convtranspose_mish_add_hardtanh_scaling_perf",
        args={},
    ))
def benchmark(label, provider):
    row = _shape_by_label(label)
    if provider in ("baseline1", "baseline2") and _direct_grid_overflows(row):
        print(f"BENCH {provider} {label} INF grid_guard")
        return float("inf")
    if provider in _MODS and _MODS[provider] is None:
        print(f"BENCH {provider} {label} INF provider_unavailable")
        return float("inf")
    x = _make_inputs(row)
    try:
        return triton.testing.do_bench(
            lambda: _run_provider(provider, row, x[0]),
            warmup=5,
            rep=20,
            return_mode="mean")
    except Exception as exc:
        print(f"BENCH {provider} {label} INF {type(exc).__name__}")
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
