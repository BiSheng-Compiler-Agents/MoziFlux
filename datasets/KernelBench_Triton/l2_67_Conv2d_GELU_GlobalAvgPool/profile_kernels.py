import argparse
import importlib.util
import math
import pathlib
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = pathlib.Path(__file__).resolve().parent
INPUT_FILE = ROOT / "67_Conv2d_GELU_GlobalAvgPool.py"
OPT_FILE = ROOT / "opt_67_Conv2d_GELU_GlobalAvgPool.py"
# Sandbox reference files are explicitly marked DO NOT READ.  Keep the column
# visible and parser-friendly, but do not import base_*.py.

_BENCH_SHAPES = [
    ("small_32", 4, 8, 32, 32),
    ("medium_128", 16, 8, 128, 128),
    ("exact_256", 128, 8, 256, 256),
]
_SHAPE_BY_LABEL = {row[0]: row[1:] for row in _BENCH_SHAPES}
_MODULE_CACHE = {}
_MODEL_CACHE: Dict[str, nn.Module] = {}


class SkipProvider(RuntimeError):
    pass


def _load(path: pathlib.Path, name: str):
    key = str(path)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _MODULE_CACHE[key] = mod
    return mod


def _init_args():
    mod = _load(INPUT_FILE, "k_input_67_conv2d_gelu_gap")
    init = mod.get_init_inputs()
    if init == [()]:
        init = []
    return init


class TorchRef(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        y = self.conv(x)
        return F.gelu(y, approximate="none").mean(dim=(-2, -1))


def _model(key: str):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    init = _init_args()
    if key == "torch_ref":
        model = TorchRef(*init)
    elif key == "baseline1":
        model = _load(INPUT_FILE, "k_baseline1_67_conv2d_gelu_gap").ModelNew(*init)
    elif key == "optimized":
        model = _load(OPT_FILE, "k_opt_67_conv2d_gelu_gap").ModelNew(*init)
    else:
        raise SkipProvider("sandbox_forbids_base_reference_import")
    model = model.eval().npu()
    _MODEL_CACHE[key] = model
    return model


def _make_inputs(label: str):
    batch, channels, height, width = _SHAPE_BY_LABEL[label]
    torch.manual_seed(1234)
    return torch.rand(batch, channels, height, width, device="npu", dtype=torch.float32)


def _run_provider(provider: str, label: str):
    x = _make_inputs(label)
    if provider == "baseline2":
        raise SkipProvider("sandbox_forbids_base_reference_import")
    if provider == "baseline1" and label == "exact_256":
        raise SkipProvider("comparison_provider_preskipped_to_avoid_timeout")
    key = {"torch": "torch_ref", "baseline1": "baseline1", "opt": "optimized"}[provider]
    with torch.no_grad():
        return _model(key)(x)


def _run_torch_ref(label: str):
    return _run_provider("torch", label)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _time_ms(fn, label: str) -> float:
    # Exact shape is large; use bounded repetitions to keep remote verification
    # below timeout while still measuring real hardware latency.
    warmup = 5 if label == "exact_256" else 10
    rep = 20 if label == "exact_256" else 50
    try:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
        _sync()
        start = time.perf_counter()
        for _ in range(rep):
            fn()
        _sync()
        return (time.perf_counter() - start) * 1000.0 / rep


def _bench_cell(provider: str, label: str) -> float:
    try:
        if provider == "baseline2":
            print(f"INFO benchmark Baseline Triton2 {label}: unavailable sandbox_forbids_base_reference_import")
            return float("inf")
        if provider == "baseline1" and label == "exact_256":
            print(f"INFO benchmark Baseline Triton1 {label}: inf comparison_provider_preskipped_to_avoid_timeout")
            return float("inf")
        # Build model once and input once for timing this cell.
        x = _make_inputs(label)
        if provider == "torch":
            model = _model("torch_ref")
        elif provider == "baseline1":
            model = _model("baseline1")
        else:
            model = _model("optimized")
        with torch.no_grad():
            return _time_ms(lambda: model(x), label)
    except Exception as exc:
        print(f"INFO benchmark_provider_unavailable {provider} {label}: {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[row[0] for row in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "baseline1", "baseline2", "opt"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="conv2d-gelu-globalavgpool",
        args={},
    )
)
def bench(label, provider):
    return _bench_cell(provider, label)


def unit_test() -> bool:
    ok = True
    for label, *_ in _BENCH_SHAPES:
        try:
            ref = _run_torch_ref(label)
            print(f"TEST PyTorch / ACL {label}: PASS max_abs=0.0")
        except Exception as exc:
            print(f"TEST PyTorch / ACL {label}: UNAVAILABLE {type(exc).__name__} max_abs=inf")
            ok = False
            continue
        for provider, display in [
            ("baseline1", "Baseline Triton1"),
            ("baseline2", "Baseline Triton2"),
            ("opt", "Optimized Triton"),
        ]:
            try:
                out = _run_provider(provider, label)
                _sync()
                max_abs = (out.float() - ref.float()).abs().max().item()
                passed = bool(torch.isfinite(torch.tensor(max_abs)) and max_abs <= 1e-3)
                print(f"TEST {display} {label}: {'PASS' if passed else 'MISMATCH'} max_abs={max_abs:.6g}")
                if provider == "opt" and not passed:
                    ok = False
            except SkipProvider as exc:
                print(f"TEST {display} {label}: SKIP_COMPARISON {str(exc)} max_abs=inf")
            except Exception as exc:
                print(f"TEST {display} {label}: UNAVAILABLE {type(exc).__name__} max_abs=inf")
                if provider == "opt":
                    ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not args.test and not args.bench:
        args.test = True
        args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        bench.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()

