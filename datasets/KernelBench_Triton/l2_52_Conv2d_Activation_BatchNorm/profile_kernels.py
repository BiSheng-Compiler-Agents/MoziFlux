import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "52_Conv2d_Activation_BatchNorm.py"
BASE_FILE = HERE / "base_52_Conv2d_Activation_BatchNorm.py"
OPT_FILE = HERE / "opt_52_Conv2d_Activation_BatchNorm.py"

# Covers optimized direct dispatch, default persistent dispatch, and irregular direct shapes.
_BENCH_SHAPES = [
    ("direct_tiny", 1, 64, 16, 16),
    ("direct_irregular", 2, 64, 31, 29),
    ("persistent_default", 64, 64, 128, 128),
]

_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_PROVIDER_NAMES = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}
_MODULES = {}
_MODELS = {}


def _load(path: Path, name: str):
    if not path.exists():
        return None
    if name in _MODULES:
        return _MODULES[name]
    try:
        spec = importlib.util.spec_from_file_location(f"k_l2_52_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MODULES[name] = mod
        return mod
    except BaseException as exc:
        print(f"INFO provider_unavailable {_PROVIDER_NAMES.get(name, name)} {type(exc).__name__}")
        _MODULES[name] = None
        return None


class TorchRef(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = x * torch.tanh(F.softplus(x, beta=1.0, threshold=20.0))
        x = self.bn(x)
        return x


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else [64, 128, 3]
    if init == [()]:
        init = []
    return init


def _model(key):
    if key in _MODELS:
        return _MODELS[key]
    torch.manual_seed(0)
    if key == "torch":
        m = TorchRef(64, 128, 3)
    else:
        path = {"baseline1": INPUT_FILE, "baseline2": BASE_FILE, "optimized": OPT_FILE}[key]
        mod = _load(path, key)
        if mod is None:
            _MODELS[key] = None
            return None
        m = mod.ModelNew(*_init_args(mod))
    m = m.eval().npu()
    _MODELS[key] = m
    return m


def _make_inputs(label):
    _, b, c, h, w = next(s for s in _BENCH_SHAPES if s[0] == label)
    torch.manual_seed(1234 + b + h + w)
    return (torch.rand((b, c, h, w), device="npu", dtype=torch.float32),)


def _activation_tiles(label):
    _, b, _, h, w = next(s for s in _BENCH_SHAPES if s[0] == label)
    out_h = h - 3 + 1
    out_w = w - 3 + 1
    n = b * 128 * out_h * out_w
    return triton.cdiv(n, 4096)


def _run_provider(key, label):
    model = _model(key)
    if model is None:
        return None
    x, = _make_inputs(label)
    # The editable baseline launches one Triton program per activation tile and exceeds Ascend coreDim at default size.
    if key == "baseline1" and _activation_tiles(label) > 65535:
        print(f"INFO comparison_provider_preskipped {_PROVIDER_NAMES[key]} {label} coreDim_guard")
        return None
    with torch.no_grad():
        return model(x)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok = True
    for label, *_ in _BENCH_SHAPES:
        ref = _run_provider("torch", label)
        opt = _run_provider("optimized", label)
        diff = _max_abs(opt, ref)
        passed = diff <= 1e-3
        print(f"CHECK optimized {label} max_abs={diff:.6g} {'PASS' if passed else 'MISMATCH'}")
        ok = ok and passed
        for key in ("baseline1", "baseline2"):
            out = _run_provider(key, label)
            if out is None:
                continue
            d = _max_abs(out, ref)
            print(f"INFO comparison { _PROVIDER_NAMES[key] } {label} max_abs={d:.6g}")

    # Unit-only forced persistent path without allocating a huge tensor.
    opt_mod = _load(OPT_FILE, "optimized")
    old = getattr(opt_mod, "_MAX_PROGRAMS", None)
    if old is not None:
        opt_mod._MAX_PROGRAMS = 1
        _MODELS.pop("optimized", None)
        ref = _run_provider("torch", "direct_irregular")
        out = _run_provider("optimized", "direct_irregular")
        d = _max_abs(out, ref)
        forced_ok = d <= 1e-3
        print(f"CHECK optimized forced_persistent direct_irregular max_abs={d:.6g} {'PASS' if forced_ok else 'MISMATCH'}")
        ok = ok and forced_ok
        opt_mod._MAX_PROGRAMS = old
        _MODELS.pop("optimized", None)

    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _manual_bench(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[_PROVIDER_NAMES[p] for p in _PROVIDERS],
        styles=[("black", "-"), ("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="Latency (ms)",
        plot_name="l2_52_conv_mish_batchnorm",
        args={},
    )
)
def bench(label, provider):
    if provider == "baseline1" and _activation_tiles(label) > 65535:
        print(f"INFO comparison_provider_preskipped {_PROVIDER_NAMES[provider]} {label} coreDim_guard")
        return float("inf")
    model = _model(provider)
    if model is None:
        return float("inf")
    x, = _make_inputs(label)

    def fn():
        with torch.no_grad():
            return model(x)

    try:
        if hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn, warmup=10, rep=30, return_mode="mean")
        return _manual_bench(fn)
    except BaseException as exc:
        print(f"INFO bench_unavailable {_PROVIDER_NAMES[provider]} {label} {type(exc).__name__}")
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
        bench.run(print_data=True, show_plots=False, save_path=str(HERE))


if __name__ == "__main__":
    main()
