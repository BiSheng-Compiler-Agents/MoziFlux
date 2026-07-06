import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = "70_Gemm_Sigmoid_Scaling_ResidualAdd.py"
OPT_FILE = "opt_70_Gemm_Sigmoid_Scaling_ResidualAdd.py"
BASE2_FILE = "base_70_Gemm_Sigmoid_Scaling_ResidualAdd.py"  # sandbox: do not read; visible skip column only

_BENCH_SHAPES = [
    ("small_irregular", 17, 257, 263),
    ("medium_aligned", 64, 1024, 1024),
    ("default", 1024, 8192, 8192),
]
_PROVIDERS = ["torch", "baseline1", "baseline2", "opt"]
_DISPLAY = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "opt": "Optimized Triton",
}
_MOD_CACHE = {}
_MODEL_CACHE = {}
_FORCE_OPT_PERSISTENT = False


def _load(filename, key):
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"k_l2_70_{key}", ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MOD_CACHE[key] = mod
    return mod


def _get_device():
    return torch.device("npu")


def _make_inputs(label, batch, input_size, hidden_size, dtype=torch.float32):
    torch.manual_seed(123)
    return (torch.rand((batch, input_size), device=_get_device(), dtype=dtype),)


def _model(key, input_size, hidden_size, scaling_factor=2.0, dtype=torch.float32):
    cache_key = (key, input_size, hidden_size, scaling_factor, str(dtype), _FORCE_OPT_PERSISTENT)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if key == "baseline2":
        raise RuntimeError("sandbox_preskip_base_read_forbidden")
    filename = INPUT_FILE if key == "baseline1" else OPT_FILE
    mod = _load(filename, key)
    torch.manual_seed(0)
    model = mod.ModelNew(input_size, hidden_size, scaling_factor).to(device=_get_device(), dtype=dtype)
    model.eval()
    if key == "opt" and _FORCE_OPT_PERSISTENT and hasattr(mod, "_MAX_GRID"):
        mod._MAX_GRID = 1
    _MODEL_CACHE[cache_key] = model
    return model


def _run_torch_ref(x, input_size, hidden_size, scaling_factor=2.0):
    model = _model("opt", input_size, hidden_size, scaling_factor, x.dtype)
    z = F.linear(x, model.gemm.weight, model.gemm.bias)
    return z + torch.sigmoid(z) * scaling_factor


def _run_provider(key, x, input_size, hidden_size, scaling_factor=2.0):
    if key == "torch":
        return _run_torch_ref(x, input_size, hidden_size, scaling_factor)
    model = _model(key, input_size, hidden_size, scaling_factor, x.dtype)
    return model(x)


def _sync():
    torch.npu.synchronize()


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def unit_test():
    ok = True
    for label, batch, input_size, hidden_size in _BENCH_SHAPES:
        x, = _make_inputs(label, batch, input_size, hidden_size)
        ref = _run_torch_ref(x, input_size, hidden_size)
        _sync()
        for key in _PROVIDERS[1:]:
            disp = _DISPLAY[key]
            if key == "baseline2":
                print(f"TEST {disp} {label}: SKIP_UNAVAILABLE sandbox_base_read_forbidden max_abs=inf")
                continue
            try:
                out = _run_provider(key, x, input_size, hidden_size)
                _sync()
                diff = _max_abs(out, ref)
                passed = diff <= 1e-3
                print(f"TEST {disp} {label}: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}")
                if key == "opt" and not passed:
                    ok = False
            except Exception as e:
                safe = type(e).__name__
                print(f"TEST {disp} {label}: {'FAIL' if key == 'opt' else 'SKIP_UNAVAILABLE'} {safe} max_abs=inf")
                if key == "opt":
                    ok = False
    global _FORCE_OPT_PERSISTENT
    try:
        _FORCE_OPT_PERSISTENT = True
        _MODEL_CACHE.clear()
        label, batch, input_size, hidden_size = ("forced_persistent", 4, 128, 128)
        x, = _make_inputs(label, batch, input_size, hidden_size)
        ref = _run_torch_ref(x, input_size, hidden_size)
        out = _run_provider("opt", x, input_size, hidden_size)
        _sync()
        diff = _max_abs(out, ref)
        passed = diff <= 1e-3
        print(f"TEST Optimized Triton {label}: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}")
        ok = ok and passed
    except Exception as e:
        print(f"TEST Optimized Triton forced_persistent: FAIL {type(e).__name__} max_abs=inf")
        ok = False
    finally:
        opt = _MOD_CACHE.get("opt")
        if opt is not None and hasattr(opt, "_MAX_GRID"):
            opt._MAX_GRID = 65535
        _FORCE_OPT_PERSISTENT = False
        _MODEL_CACHE.clear()
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_once(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / rep


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[_DISPLAY[p] for p in _PROVIDERS],
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="latency_ms",
        plot_name="l2_70_gemm_sigmoid_scaling_residualadd",
        args={},
    )
)
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, batch, input_size, hidden_size = shape
    if provider == "baseline2":
        print(f"INFO benchmark_preskip {_DISPLAY[provider]} {label}: sandbox_base_read_forbidden")
        return float("inf")
    try:
        x, = _make_inputs(label, batch, input_size, hidden_size)
        fn = lambda: _run_provider(provider, x, input_size, hidden_size)
        return _bench_once(fn)
    except Exception as e:
        print(f"INFO benchmark_unavailable {_DISPLAY[provider]} {label}: {type(e).__name__}")
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
        bench.run(print_data=True, show_plots=False, save_path=str(ROOT / "profile_plots"))


if __name__ == "__main__":
    main()
