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

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "4_Conv2d_Mish_Mish.py"
BASE_FILE = ROOT / "base_4_Conv2d_Mish_Mish.py"
OPT_FILE = ROOT / "opt_4_Conv2d_Mish_Mish.py"

_BENCH_SHAPES = [
    ("small_direct", 1, 64, 32, 32),
    ("medium_direct", 4, 64, 64, 64),
    ("default_shape", 64, 64, 256, 256),
]
_SHAPES = {s[0]: s[1:] for s in _BENCH_SHAPES}
_PROVIDERS = ["torch", "baseline1", "baseline2", "opt"]
_PROVIDER_NAMES = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "opt": "Optimized Triton",
}
_MODEL_CACHE = {}
_MODULE_CACHE = {}
_INIT = [64, 128, 3]


def _load(path: Path, key: str):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    try:
        spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MODULE_CACHE[key] = mod
        return mod
    except Exception as exc:
        print(f"INFO provider_unavailable {key}: {type(exc).__name__}")
        _MODULE_CACHE[key] = None
        return None


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _make_inputs(label):
    shape = _SHAPES[label]
    torch.manual_seed(123)
    return (torch.rand(shape, device="npu", dtype=torch.float32),)


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    if key == "torch":
        model = nn.Conv2d(*_INIT).npu().eval()
    else:
        path = {"baseline1": INPUT_FILE, "baseline2": BASE_FILE, "opt": OPT_FILE}[key]
        mod = _load(path, key)
        if mod is None:
            _MODEL_CACHE[key] = None
            return None
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else _INIT
        if init == [()]:
            init = []
        model = mod.ModelNew(*init).npu().eval()
    _MODEL_CACHE[key] = model
    return model


def _run_torch_ref(x):
    with torch.no_grad():
        y = _model("torch")(x)
        return F.mish(F.mish(y))


def _run_provider(key, x):
    if key == "torch":
        return _run_torch_ref(x)
    model = _model(key)
    if model is None:
        raise RuntimeError(f"provider {key} unavailable")
    with torch.no_grad():
        return model(x)


def _max_abs(a, b):
    return (a - b).abs().max().detach().float().cpu().item()


def unit_test():
    ok = True
    for label, *_ in _BENCH_SHAPES:
        (x,) = _make_inputs(label)
        ref = _run_torch_ref(x)
        _sync()
        for key in _PROVIDERS[1:]:
            name = _PROVIDER_NAMES[key]
            try:
                out = _run_provider(key, x)
                _sync()
                diff = _max_abs(out, ref)
                if diff <= 1e-3:
                    print(f"PASS {name} {label} max_abs={diff:.6g}")
                elif key == "opt":
                    print(f"UNIT_TEST_FAILED {name} {label} max_abs={diff:.6g}")
                    ok = False
                else:
                    print(f"INFO comparison_mismatch {name} {label} max_abs={diff:.6g}")
            except Exception as exc:
                if key == "opt":
                    print(f"UNIT_TEST_FAILED {name} {label}: {type(exc).__name__}")
                    ok = False
                else:
                    print(f"INFO comparison_provider_unavailable {name} {label}: {type(exc).__name__}")
    # Force custom Triton direct and persistent paths; production dispatch uses ACL.
    try:
        opt_mod = _load(OPT_FILE, "opt")
        old_cap = getattr(opt_mod, "_MAX_GRID", None)
        old_acl = getattr(opt_mod, "_USE_ACL_DISPATCH", True)
        opt_mod._USE_ACL_DISPATCH = False
        for forced_label, cap, shape_label in [
            ("forced_direct", 65535, "small_direct"),
            ("forced_persistent", 1, "medium_direct"),
        ]:
            opt_mod._MAX_GRID = cap
            _MODEL_CACHE.pop("opt", None)
            (x,) = _make_inputs(shape_label)
            ref = _run_torch_ref(x)
            out = _run_provider("opt", x)
            _sync()
            diff = _max_abs(out, ref)
            if diff <= 1e-3:
                print(f"PASS Optimized Triton {forced_label} max_abs={diff:.6g}")
            else:
                print(f"UNIT_TEST_FAILED Optimized Triton {forced_label} max_abs={diff:.6g}")
                ok = False
        if old_cap is not None:
            opt_mod._MAX_GRID = old_cap
        opt_mod._USE_ACL_DISPATCH = old_acl
        _MODEL_CACHE.pop("opt", None)
    except Exception as exc:
        print(f"UNIT_TEST_FAILED Optimized Triton forced_paths: {type(exc).__name__}")
        ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_once(provider, label):
    (x,) = _make_inputs(label)
    try:
        fn = lambda: _run_provider(provider, x)
        # Avoid poisoning the process with providers that are known to be unavailable.
        if provider != "opt" and _model(provider) is None:
            print(f"INFO benchmark_provider_unavailable {_PROVIDER_NAMES[provider]} {label}")
            return float("inf")
        return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="mean")
    except Exception:
        print(f"INFO benchmark_provider_unavailable {_PROVIDER_NAMES[provider]} {label}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[_PROVIDER_NAMES[p] for p in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "-")],
        ylabel="ms",
        plot_name="conv2d_mish_mish",
        args={},
    )
)
def benchmark(label, provider):
    return _bench_once(provider, label)


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
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
