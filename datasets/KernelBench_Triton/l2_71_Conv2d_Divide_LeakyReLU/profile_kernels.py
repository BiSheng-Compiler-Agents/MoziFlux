import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = "71_Conv2d_Divide_LeakyReLU.py"
OPT_FILE = "opt_71_Conv2d_Divide_LeakyReLU.py"

_BENCH_SHAPES = [
    ("small", 8, 8, 31, 33),
    ("irregular", 17, 8, 65, 67),
    ("default", 128, 8, 128, 128),
]
_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_DISPLAY = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}
_MODULE_CACHE = {}
_MODEL_CACHE = {}
_BASELINE2_SKIP = "sandbox_reference_file_do_not_read"


class TorchRef(nn.Module):
    def __init__(self, in_channels=8, out_channels=64, kernel_size=3, divisor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        y = self.conv(x)
        return F.leaky_relu(y / self.divisor, negative_slope=0.01)


def _load(filename, key):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def _init_args_from_input():
    mod = _load(INPUT_FILE, "input")
    init = mod.get_init_inputs()
    if init == [()]:
        init = []
    return list(init)


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    init = _init_args_from_input()
    if key == "torch":
        model = TorchRef(*init)
    elif key == "baseline1":
        model = _load(INPUT_FILE, "baseline1").ModelNew(*init)
    elif key == "optimized":
        model = _load(OPT_FILE, "optimized").ModelNew(*init)
    else:
        raise RuntimeError(_BASELINE2_SKIP)
    model = model.to("npu").eval()
    _MODEL_CACHE[key] = model
    return model


def _make_input(label):
    for item in _BENCH_SHAPES:
        if item[0] == label:
            _, batch, channels, height, width = item
            torch.manual_seed(1234 + batch + height + width)
            return torch.rand(batch, channels, height, width, device="npu")
    raise KeyError(label)


def _run_provider(provider, x):
    with torch.no_grad():
        return _model(provider)(x)


def _sync():
    torch.npu.synchronize()


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def _check_one(provider, label):
    x = _make_input(label)
    ref = _run_provider("torch", x)
    _sync()
    if provider == "baseline2":
        print(f"TEST {_DISPLAY[provider]} {label}: SKIP_UNAVAILABLE {_BASELINE2_SKIP} max_abs=inf")
        return True
    try:
        out = _run_provider(provider, x)
        _sync()
        diff = _max_abs(out, ref)
        ok = math.isfinite(diff) and diff <= 1e-3
        status = "PASS" if ok else "MISMATCH"
        print(f"TEST {_DISPLAY[provider]} {label}: {status} max_abs={diff:.6g}")
        return ok or provider != "optimized"
    except Exception as exc:
        etype = type(exc).__name__
        if provider == "optimized":
            print(f"TEST {_DISPLAY[provider]} {label}: MISMATCH {etype} max_abs=inf")
            return False
        print(f"TEST {_DISPLAY[provider]} {label}: SKIP_UNAVAILABLE {etype} max_abs=inf")
        return True


def unit_test():
    ok = True
    for label, *_ in _BENCH_SHAPES:
        # Always construct reference first, then comparison providers.
        _check_one("torch", label)
        for provider in ["baseline1", "baseline2", "optimized"]:
            if not _check_one(provider, label):
                ok = False
    if not _forced_persistent_test():
        ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _forced_persistent_test():
    try:
        opt = _load(OPT_FILE, "optimized")
        old = getattr(opt, "_MAX_GRID", 65535)
        opt._MAX_GRID = 1
        _MODEL_CACHE.pop("optimized", None)
        label = "small"
        x = _make_input(label)
        ref = _run_provider("torch", x)
        out = _run_provider("optimized", x)
        _sync()
        diff = _max_abs(out, ref)
        ok = math.isfinite(diff) and diff <= 1e-3
        print(f"TEST Optimized Triton forced_persistent: {'PASS' if ok else 'MISMATCH'} max_abs={diff:.6g}")
        return ok
    except Exception as exc:
        print(f"TEST Optimized Triton forced_persistent: MISMATCH {type(exc).__name__} max_abs=inf")
        return False
    finally:
        try:
            opt._MAX_GRID = old
            _MODEL_CACHE.pop("optimized", None)
        except Exception:
            pass


def _time_ms(fn, warmup=3, rep=10):
    for _ in range(max(1, warmup)):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(max(1, rep)):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / max(1, rep)


def _bench_provider(provider, label):
    if provider == "baseline2":
        print(f"INFO benchmark_preskip {_DISPLAY[provider]} {label}: {_BASELINE2_SKIP}")
        return float("inf")
    x = _make_input(label)
    try:
        _run_provider(provider, x)
        _sync()
        return _time_ms(lambda: _run_provider(provider, x))
    except Exception as exc:
        print(f"INFO benchmark_unavailable {_DISPLAY[provider]} {label}: {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[_DISPLAY[p] for p in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="conv2d_divide_leakyrelu",
        args={},
    )
)
def benchmark(label, provider):
    return _bench_provider(provider, label)


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
        benchmark.run(print_data=True, show_plots=False, save_path=str(ROOT / "remote_results"))


if __name__ == "__main__":
    main()
