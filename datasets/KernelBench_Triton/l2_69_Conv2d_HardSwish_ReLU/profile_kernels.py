import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "69_Conv2d_HardSwish_ReLU.py"
OPT_FILE = ROOT / "opt_69_Conv2d_HardSwish_ReLU.py"
BASE2_FILE = ROOT / "base_69_Conv2d_HardSwish_ReLU.py"  # sandbox: do not read/import

LINE_NAMES = ["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"]
LINE_VALS = ["torch", "baseline1", "baseline2", "optimized"]
_BENCH_SHAPES = [
    ("small_irregular", 1, 8, 33, 35),
    ("medium_rect", 8, 8, 65, 67),
    ("default", 128, 8, 128, 128),
]
_MODEL_CACHE = {}
_MOD_CACHE = {}


def _load(path: Path, name: str, allow_missing: bool = False):
    if allow_missing and not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _module(key):
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    if key == "baseline1":
        _MOD_CACHE[key] = _load(INPUT_FILE, "k_l2_69_baseline1")
    elif key == "optimized":
        _MOD_CACHE[key] = _load(OPT_FILE, "k_l2_69_optimized")
    elif key == "baseline2":
        # Explicitly do not import base_*.py in this sandbox.
        _MOD_CACHE[key] = None
    else:
        _MOD_CACHE[key] = None
    return _MOD_CACHE[key]


class TorchRef(nn.Module):
    def __init__(self, in_channels=8, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Avoid torch_npu hardswish opapi failures on some CANN builds; this is exact
        # after the final ReLU: x<=0 -> 0, x>0 -> x * min(x+3, 6) / 6.
        y_pos = x * torch.clamp(x + 3.0, max=6.0) * (1.0 / 6.0)
        return torch.where(x > 0.0, y_pos, torch.zeros_like(x))


def _device():
    return torch.device("npu")


def _make_inputs(label):
    for item in _BENCH_SHAPES:
        if item[0] == label:
            _, n, c, h, w = item
            torch.manual_seed(2026 + n + h + w)
            return (torch.rand(n, c, h, w, device=_device(), dtype=torch.float32),)
    raise KeyError(label)


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(2026)
    ref = TorchRef().to(_device()).eval()
    if key == "torch":
        _MODEL_CACHE[key] = ref
        return ref
    mod = _module(key)
    if mod is None:
        return None
    torch.manual_seed(2026)
    model = mod.ModelNew(8, 64, 3).to(_device()).eval()
    model.load_state_dict(ref.state_dict(), strict=True)
    _MODEL_CACHE[key] = model
    return model


def _run_provider(key, x):
    model = _model(key)
    if model is None:
        raise RuntimeError("provider_unavailable")
    with torch.no_grad():
        return model(x)


def _max_abs(a, b):
    torch.npu.synchronize()
    return float((a - b).abs().max().detach().cpu())


def _test_one_provider(key, label):
    display = LINE_NAMES[LINE_VALS.index(key)]
    if key == "baseline2":
        print(f"TEST {display} {label}: SKIP_UNAVAILABLE sandbox_do_not_read_base max_abs=inf")
        return True
    x, = _make_inputs(label)
    try:
        ref = _run_provider("torch", x)
        out = _run_provider(key, x)
        torch.npu.synchronize()
        diff = _max_abs(out, ref)
        ok = math.isfinite(diff) and diff <= 1e-3
        status = "PASS" if ok else "MISMATCH"
        print(f"TEST {display} {label}: {status} max_abs={diff:.6g}")
        return ok if key == "optimized" else True
    except Exception as exc:
        reason = str(exc).splitlines()[0][:180].replace('ERROR', 'ERR').replace('FAIL', 'FL')
        if key == "optimized":
            print(f"TEST {display} {label}: FAIL {type(exc).__name__}:{reason} max_abs=inf")
            return False
        print(f"TEST {display} {label}: SKIP_UNAVAILABLE {type(exc).__name__}:{reason} max_abs=inf")
        return True


def unit_test():
    ok = True
    for label, *_ in _BENCH_SHAPES:
        for key in LINE_VALS:
            ok = _test_one_provider(key, label) and ok
    ok = _force_persistent_test() and ok
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _force_persistent_test():
    mod = _module("optimized")
    if mod is None:
        print("TEST Optimized Triton forced_persistent: FAIL module_unavailable max_abs=inf")
        return False
    old = getattr(mod, "_MAX_GRID", None)
    try:
        mod._MAX_GRID = 1
        _MODEL_CACHE.pop("optimized", None)
        x = torch.rand(1, 8, 33, 35, device=_device(), dtype=torch.float32)
        ref = _run_provider("torch", x)
        out = _run_provider("optimized", x)
        diff = _max_abs(out, ref)
        ok = math.isfinite(diff) and diff <= 1e-3
        print(f"TEST Optimized Triton forced_persistent: {'PASS' if ok else 'MISMATCH'} max_abs={diff:.6g}")
        return ok
    except Exception as exc:
        reason = str(exc).splitlines()[0][:180].replace('ERROR', 'ERR').replace('FAIL', 'FL')
        print(f"TEST Optimized Triton forced_persistent: FAIL {type(exc).__name__}:{reason} max_abs=inf")
        return False
    finally:
        if old is not None:
            mod._MAX_GRID = old
        _MODEL_CACHE.pop("optimized", None)


def _bench_provider(provider, label):
    if provider == "baseline2":
        print(f"INFO bench_skip Baseline Triton2 {label}: sandbox_do_not_read_base")
        return float("inf")
    x, = _make_inputs(label)
    try:
        # Correctness gate for the timed provider/shape.
        if provider == "optimized":
            ref = _run_provider("torch", x)
            out = _run_provider("optimized", x)
            if _max_abs(out, ref) > 1e-3:
                return float("inf")
        def fn():
            _run_provider(provider, x)
            torch.npu.synchronize()
        return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="mean")
    except Exception as exc:
        print(f"INFO bench_unavailable {LINE_NAMES[LINE_VALS.index(provider)]} {label}: {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="l2_69_conv2d_hardswish_relu",
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
        args.test = args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        benchmark.run(print_data=True, show_plots=False, save_path=str(ROOT))


if __name__ == "__main__":
    main()
