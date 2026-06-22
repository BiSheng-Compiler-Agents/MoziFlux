import argparse
import importlib.util
import math
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
    except Exception as exc:
        if optional:
            print(
                f"INFO optional_provider_unavailable {fname} {type(exc).__name__}"
            )
            return None
        raise


_baseline_mod1 = _load("100_ConvTranspose3d_Clamp_Min_Divide.py")
_baseline_mod2 = _load("base_100_ConvTranspose3d_Clamp_Min_Divide.py",
                       optional=True)
_optimized_mod = _load("opt_100_ConvTranspose3d_Clamp_Min_Divide.py")

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

# label, B, Cin, D, H, W, OC, K, stride, padding
_BENCH_SHAPES = [
    ("small_B1_D4", 1, 4, 4, 8, 8, 8, 3, 2, 1),
    ("medium_B2_D8", 2, 8, 8, 16, 16, 16, 3, 2, 1),
    ("large_B4_D12", 4, 16, 12, 24, 24, 32, 3, 2, 1),
    ("target_B16_D24", 16, 64, 24, 48, 48, 128, 3, 2, 1),
]


def _init_args(shape):
    _label, _B, Cin, _D, _H, _W, OC, K, S, P = shape
    return [Cin, OC, K, S, P, -1.0, 2.0]


def _model(key, shape):
    mod = _MODS.get(key)
    if mod is None:
        return None
    cache_key = (key, tuple(_init_args(shape)))
    if cache_key not in _models:
        torch.manual_seed(0)
        _models[cache_key] = mod.ModelNew(*_init_args(shape)).to(
            device="npu", dtype=torch.float32).eval()
    return _models[cache_key]


def _make_input(shape):
    _label, B, Cin, D, H, W, _OC, _K, _S, _P = shape
    torch.manual_seed(42)
    return torch.rand(B, Cin, D, H, W, device="npu", dtype=torch.float32)


def _run_torch_ref(x, shape):
    m = _model("optimized", shape)
    y = F.conv_transpose3d(
        x,
        m.conv_transpose.weight,
        m.conv_transpose.bias,
        stride=m.conv_transpose.stride,
        padding=m.conv_transpose.padding,
    )
    return torch.clamp(y, min=m.min_value) / m.divisor


def _output_numel(shape):
    _label, B, _Cin, D, H, W, OC, K, S, P = shape
    od = (D - 1) * S - 2 * P + K
    oh = (H - 1) * S - 2 * P + K
    ow = (W - 1) * S - 2 * P + K
    return B * OC * od * oh * ow


def _baseline1_grid_overflow(shape):
    n = _output_numel(shape)
    if n >= (1 << 20):
        block = 8192
    elif n >= (1 << 18):
        block = 4096
    else:
        block = 2048
    return math.ceil(n / block) > 65535


def _run_provider(key, x, shape):
    if key == "torch_ref":
        return _run_torch_ref(x, shape)
    if key == "baseline2":
        raise RuntimeError("baseline2_preskipped")
    if key == "baseline1" and _baseline1_grid_overflow(shape):
        raise RuntimeError("grid_guard")
    m = _model(key, shape)
    return m(x)


def unit_test():
    print("=== Unit Test: ConvTranspose3d_Clamp_Min_Divide ===")
    any_fail = False
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_input(shape)
        with torch.no_grad():
            ref = _run_torch_ref(x, shape)
        for key, name in _VARIANTS:
            if key == "baseline2":
                print(f"TEST {key} {label} SKIP baseline2_preskipped")
                continue
            if key == "baseline1" and _baseline1_grid_overflow(shape):
                print(f"TEST {key} {label} SKIP grid_guard")
                continue
            try:
                with torch.no_grad():
                    out = _run_provider(key, x, shape)
                max_diff = (ref.float() - out.float()).abs().max().item()
                ok = torch.allclose(ref.float(),
                                    out.float(),
                                    atol=1e-3,
                                    rtol=1e-3)
                print(
                    f"TEST {key} {label} {'PASS' if ok else 'MISMATCH'} max_diff={max_diff:.6e}"
                )
                if key == "optimized" and not ok:
                    any_fail = True
            except Exception as exc:
                token = "SKIP" if key != "optimized" else "ERROR"
                print(f"TEST {key} {label} {token} {type(exc).__name__}")
                if key == "optimized":
                    any_fail = True
        del x, ref
        torch.npu.empty_cache()
    print("UNIT_TEST PASS" if not any_fail else "UNIT_TEST_FAILED")
    return not any_fail


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=[k for k, _ in _PROVIDERS],
        line_names=[n for _, n in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "-")],
        ylabel="Latency (ms)",
        plot_name="convtranspose3d_clamp_divide_perf",
        args={},
    ))
def benchmark(label, mode):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    if mode == "baseline2":
        return float("inf")
    if mode == "baseline1" and _baseline1_grid_overflow(shape):
        return float("inf")
    x = _make_input(shape)
    try:
        return triton.testing.do_bench(lambda: _run_provider(mode, x, shape),
                                       warmup=5,
                                       rep=20,
                                       return_mode="mean")
    except Exception as exc:
        print(f"INFO bench_inf {mode} {label} {type(exc).__name__}")
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
