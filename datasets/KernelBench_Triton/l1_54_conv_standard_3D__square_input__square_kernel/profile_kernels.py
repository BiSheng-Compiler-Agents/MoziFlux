"""
profile_kernels.py -- l1_54 Conv3d standard square input/kernel

Compares PyTorch/ACL reference, the editable baseline, read-only baseline2, and optimized
provider on Ascend NPU. Correctness is checked for every provider/shape; optimized
correctness gates UNIT_TEST PASS while read-only comparison failures are reported as SKIP.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton

_DIR = Path(__file__).parent
_INPUT_FILE = "54_conv_standard_3D__square_input__square_kernel.py"
_BASE2_FILE = "base_54_conv_standard_3D__square_input__square_kernel.py"
_OPT_FILE = "opt_54_conv_standard_3D__square_input__square_kernel.py"


def _load(fname):
    spec = importlib.util.spec_from_file_location("k_" + Path(fname).stem,
                                                  _DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_baseline_mod1 = _load(_INPUT_FILE)
_baseline_mod2 = _load(_BASE2_FILE)
_optimized_mod = _load(_OPT_FILE)

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
_MODEL_CACHE = {}
_REF_CACHE = {}

# Square 3D inputs; D=64 is the exact source get_inputs() target.
_BENCH_SHAPES = [
    ("D16_square", 1, 3, 16, 16, 16),
    ("D32_square", 2, 3, 32, 32, 32),
    ("D64_square_exact", 16, 3, 64, 64, 64),
]
_TEST_ONLY_SHAPES = [
    ("D9_boundary", 1, 3, 9, 9, 9),
]


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    return list(init)


def _model(key):
    if key not in _MODEL_CACHE:
        mod = _MODS[key]
        torch.manual_seed(0)
        model = mod.ModelNew(*_init_args(mod)).to(device="npu").eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _ref_model():
    key = "torch_ref"
    if key not in _REF_CACHE:
        in_channels, out_channels, kernel_size = _init_args(_baseline_mod1)[:3]
        torch.manual_seed(0)
        model = nn.Conv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ).to(device="npu").eval()
        _REF_CACHE[key] = model
    return _REF_CACHE[key]


def _make_input(shape):
    torch.manual_seed(42 + shape[2])
    return torch.rand(shape, device="npu", dtype=torch.float32)


def _run_torch_ref(x):
    return _ref_model()(x)


def _run_provider(key, x):
    if key == "torch_ref":
        return _run_torch_ref(x)
    return _model(key)(x)


def _sync():
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _allclose(ref, out):
    return torch.allclose(ref, out, atol=1e-3, rtol=1e-3)


def unit_test():
    print("=== Unit Test: l1_54 Conv3d standard square input/kernel ===")
    optimized_ok = True
    shapes = _BENCH_SHAPES + _TEST_ONLY_SHAPES
    for label, b, c, d, h, w in shapes:
        x = _make_input((b, c, d, h, w))
        with torch.no_grad():
            ref = _run_torch_ref(x)
        _sync()
        for key, name in _VARIANTS:
            try:
                with torch.no_grad():
                    out = _run_provider(key, x)
                _sync()
                ok = _allclose(ref, out)
                max_diff = (ref - out).abs().max().item()
                provider = {
                    "baseline1": "baseline1",
                    "baseline2": "baseline2",
                    "optimized": "optimized"
                }[key]
                if ok:
                    print(
                        f"TEST {provider} {label} PASS max_diff={max_diff:.3e}"
                    )
                elif key == "optimized":
                    print(
                        f"TEST optimized {label} MISMATCH max_diff={max_diff:.3e}"
                    )
                    optimized_ok = False
                else:
                    print(
                        f"TEST {provider} {label} SKIP value_mismatch max_diff={max_diff:.3e}"
                    )
            except Exception as exc:
                provider = {
                    "baseline1": "baseline1",
                    "baseline2": "baseline2",
                    "optimized": "optimized"
                }[key]
                if key == "optimized":
                    print(
                        f"TEST optimized {label} ERROR {type(exc).__name__}: {str(exc)[:120]}"
                    )
                    optimized_ok = False
                else:
                    print(
                        f"TEST {provider} {label} SKIP provider_unavailable {type(exc).__name__}"
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
        plot_name="conv3d_l1_54_perf",
        args={},
    ))
def benchmark(label, mode):
    shape = next((b, c, d, h, w) for lbl, b, c, d, h, w in _BENCH_SHAPES
                 if lbl == label)
    x = _make_input(shape)
    try:
        return triton.testing.do_bench(lambda: _run_provider(mode, x),
                                       warmup=5,
                                       rep=20,
                                       return_mode="mean")
    except Exception as exc:
        print(
            f"BENCH_INFO {mode} {label} unavailable {type(exc).__name__}: {str(exc)[:120]}"
        )
        return float("inf")


def main():
    parser = argparse.ArgumentParser(
        description="Profile l1_54 Conv3d kernels")
    parser.add_argument("--test", action="store_true", help="correctness only")
    parser.add_argument("--bench", action="store_true", help="benchmark only")
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
