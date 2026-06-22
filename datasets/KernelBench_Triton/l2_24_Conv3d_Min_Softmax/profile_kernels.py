import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

HERE = Path(__file__).resolve().parent


def _load(fname, name):
    path = HERE / fname
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, name):
    try:
        return _load(fname, name)
    except Exception as exc:
        print(
            f"INFO optional_provider {name} unavailable {type(exc).__name__}")
        return None


_baseline1 = _load("24_Conv3d_Min_Softmax.py",
                   "k_baseline1_24_conv3d_min_softmax")
_baseline2 = _load_optional("base_24_Conv3d_Min_Softmax.py",
                            "k_baseline2_24_conv3d_min_softmax")
_optimized = _load("opt_24_Conv3d_Min_Softmax.py",
                   "k_optimized_24_conv3d_min_softmax")

_BENCH_SHAPES = [
    ("small", 4, 3, 8, 12, 12),
    ("medium", 16, 3, 16, 24, 24),
    ("target", 128, 3, 24, 32, 32),
]

_PROVIDER_NAMES = {
    "torch_ref": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}
_MODEL_CACHE = {}


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _init_args(out_channels=24):
    return [3, out_channels, 3, 2]


def _make_inputs(label, B, C, D, H, W):
    torch.manual_seed(1234)
    return [torch.rand((B, C, D, H, W), device="npu", dtype=torch.float32)]


def _make_torch_ref(out_channels=24):
    torch.manual_seed(0)
    conv = nn.Conv3d(3, out_channels, 3).to(device="npu").eval()
    return conv


def _model(provider, out_channels=24):
    key = (provider, out_channels)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if provider == "baseline2" and _baseline2 is None:
        return None
    mod = {
        "baseline1": _baseline1,
        "baseline2": _baseline2,
        "optimized": _optimized
    }[provider]
    torch.manual_seed(0)
    model = mod.ModelNew(*_init_args(out_channels)).to(device="npu").eval()
    _MODEL_CACHE[key] = model
    return model


def _run_torch_ref(x, out_channels=24):
    conv = _make_torch_ref(out_channels)
    with torch.no_grad():
        y = conv(x).contiguous()
        return F.softmax(torch.amin(y, dim=2), dim=1)


def _run_provider(provider, x, out_channels=24):
    if provider == "torch_ref":
        return _run_torch_ref(x, out_channels)
    model = _model(provider, out_channels)
    if model is None:
        raise RuntimeError("provider_unavailable")
    with torch.no_grad():
        return model(x)


def _max_abs(a, b):
    return (a - b).abs().max().item()


def _test_one(provider, label, dims, out_channels=24, required=True):
    x = _make_inputs(label, *dims)[0]
    try:
        ref = _run_torch_ref(x, out_channels)
        _sync()
        out = _run_provider(provider, x, out_channels)
        _sync()
        diff = _max_abs(out, ref)
        ok = bool(torch.allclose(out, ref, rtol=1e-3, atol=1e-3))
        status = "PASS" if ok else "BAD"
        print(f"TEST {provider} {label} {status} max_abs={diff:.6e}")
        return ok or (not required)
    except Exception as exc:
        if required:
            print(
                f"TEST {provider} {label} BAD exception={type(exc).__name__}")
            return False
        print(f"TEST {provider} {label} SKIP provider_unavailable")
        return True


def unit_test():
    all_ok = True
    for label, B, C, D, H, W in _BENCH_SHAPES:
        dims = (B, C, D, H, W)
        for provider in ("baseline1", "baseline2", "optimized"):
            required = provider == "optimized" or provider == "baseline1"
            if provider == "baseline2":
                required = False
            all_ok = _test_one(provider, label, dims, 24,
                               required=required) and all_ok
    # dispatch-only coverage for optimized C>64 ACL fallback, kept small.
    all_ok = _test_one(
        "optimized", "fallback_c80",
        (1, 3, 6, 8, 8), 80, required=True) and all_ok
    print("UNIT_TEST PASS" if all_ok else "UNIT_TEST_FAILED")
    return all_ok


def _bench_provider(provider, label, B, C, D, H, W):
    if provider == "baseline2" and _baseline2 is None:
        return float("inf")
    x = _make_inputs(label, B, C, D, H, W)[0]
    try:
        # Guard by correctness before timing this exact cell.
        ref = _run_torch_ref(x)
        out = _run_provider(provider, x)
        _sync()
        if not torch.allclose(out, ref, rtol=1e-3, atol=1e-3):
            print(f"INFO bench_preskip {provider} {label} value_mismatch")
            return float("inf")

        def fn():
            _run_provider(provider, x)

        return triton.testing.do_bench(fn,
                                       warmup=5,
                                       rep=20,
                                       return_mode="mean")
    except Exception as exc:
        print(f"INFO bench_preskip {provider} {label} {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="conv3d_min_softmax",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, B, C, D, H, W = shape
    return _bench_provider(provider, label, B, C, D, H, W)


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
        bench.run(print_data=True, show_plots=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
