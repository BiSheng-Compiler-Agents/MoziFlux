import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py"
BASE_FILE = ROOT / "base_46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py"
OPT_FILE = ROOT / "opt_46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_MODS = {}
_LOAD_ERRORS = {}
for key, path in [("baseline1", INPUT_FILE), ("baseline2", BASE_FILE), ("optimized", OPT_FILE)]:
    try:
        _MODS[key] = _load(path, f"k46_{key}_{path.stem}")
    except Exception as exc:
        _LOAD_ERRORS[key] = type(exc).__name__


class TorchRef(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        x = self.conv(x.to(dtype=torch.float32))
        return self.avgpool(torch.tanh(x - self.subtract1_value)) - self.subtract2_value


_MODEL_CACHE = {}


def _init_args():
    init = _MODS["optimized"].get_init_inputs()
    if init == [()]:
        init = []
    return init


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    if key == "torch_ref":
        model = TorchRef(*_init_args())
    elif key == "opt_triton_direct":
        model = _MODS["optimized"].ModelNew(*_init_args(), use_triton_epilogue=True)
        _MODS["optimized"]._MAX_GRID = 10**9
    elif key == "opt_triton_persistent":
        model = _MODS["optimized"].ModelNew(*_init_args(), use_triton_epilogue=True)
        _MODS["optimized"]._MAX_GRID = 1
    else:
        model = _MODS[key].ModelNew(*_init_args())
    model = model.to(device="npu", dtype=torch.float32).eval()
    _MODEL_CACHE[key] = model
    return model


_BENCH_SHAPES = [
    ("small_direct", 4, 64, 32, 32),
    ("irregular_direct", 2, 64, 35, 37),
    ("default_persistent_boundary", 128, 64, 128, 128),
]


def _make_inputs(label, B, C, H, W):
    torch.manual_seed(123)
    return (torch.rand(B, C, H, W, device="npu", dtype=torch.float32),)


def _run_torch_ref(x):
    with torch.no_grad():
        return _model("torch_ref")(x)


def _run_provider(provider, x):
    if provider in _LOAD_ERRORS:
        raise RuntimeError(f"provider_unavailable:{provider}:{_LOAD_ERRORS[provider]}")
    with torch.no_grad():
        return _model(provider)(x)


def _run_optimized_triton_path(path_key, x):
    with torch.no_grad():
        return _model(path_key)(x)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok = True
    providers = ["baseline1", "baseline2", "optimized"]
    for label, B, C, H, W in _BENCH_SHAPES:
        x, = _make_inputs(label, B, C, H, W)
        ref = _run_torch_ref(x)
        for provider in providers:
            try:
                out = _run_provider(provider, x)
                diff = _max_abs(out, ref)
                passed = diff <= 1e-3
                print(f"UNIT {label} {provider}: max_abs={diff:.6g} {'PASS' if passed else 'MISMATCH'}")
                if provider == "optimized" and not passed:
                    ok = False
            except Exception as exc:
                tag = "INFO" if provider == "baseline2" else "ERROR"
                print(f"{tag} {label} {provider}: {type(exc).__name__}")
                if provider == "optimized":
                    ok = False
    # Explicitly exercise both optimized Triton dispatch paths without making them the production default.
    for path_key, shape in [("opt_triton_direct", ("forced_direct", 2, 64, 16, 16)), ("opt_triton_persistent", ("forced_persistent", 2, 64, 16, 16))]:
        label, B, C, H, W = shape
        try:
            x, = _make_inputs(label, B, C, H, W)
            ref = _run_torch_ref(x)
            out = _run_optimized_triton_path(path_key, x)
            diff = _max_abs(out, ref)
            passed = diff <= 2e-3
            print(f"UNIT {label} {path_key}: max_abs={diff:.6g} {'PASS' if passed else 'MISMATCH'}")
            ok = ok and passed
        except Exception as exc:
            print(f"ERROR {label} {path_key}: {type(exc).__name__}")
            ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_ms(fn, warmup=10, rep=50):
    try:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / rep


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="conv2d_subtract_tanh_subtract_avgpool",
        args={},
    )
)
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    x, = _make_inputs(*shape)
    if provider in _LOAD_ERRORS:
        print(f"INFO {provider} unavailable_or_preskipped {_LOAD_ERRORS[provider]}")
        return float("inf")
    try:
        if provider == "torch_ref":
            return _bench_ms(lambda: _run_torch_ref(x))
        return _bench_ms(lambda: _run_provider(provider, x))
    except Exception as exc:
        print(f"INFO {provider} benchmark_unavailable {type(exc).__name__}")
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
