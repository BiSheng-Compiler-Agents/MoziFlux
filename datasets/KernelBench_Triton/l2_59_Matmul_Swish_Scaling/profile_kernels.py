import argparse
import importlib.util
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = pathlib.Path(__file__).resolve().parent
INPUT_FILE = ROOT / "59_Matmul_Swish_Scaling.py"
OPT_FILE = ROOT / "opt_59_Matmul_Swish_Scaling.py"
BASE2_FILE = ROOT / "base_59_Matmul_Swish_Scaling.py"

_BENCH_SHAPES = [
    ("small_4x1024x1024", 4, 1024, 1024),
    ("medium_16x4096x4096", 16, 4096, 4096),
    ("default_128x32768x32768", 128, 32768, 32768),
]
_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_MODEL_CACHE = {}
_MODULE_CACHE = {}


def _load(path: pathlib.Path, key: str):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def _init_args(batch, in_features, out_features, scale):
    return [in_features, out_features, scale]


def _make_input(batch, in_features, dtype=torch.float16):
    torch.manual_seed(123)
    return torch.rand(batch, in_features, device="npu", dtype=dtype)


def _torch_model(batch, in_features, out_features, scale, dtype):
    key = ("torch", in_features, out_features, scale, str(dtype))
    if key not in _MODEL_CACHE:
        torch.manual_seed(0)
        m = nn.Linear(in_features, out_features).eval().to("npu").to(dtype)
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


def _model(provider, batch, in_features, out_features, scale, dtype):
    key = (provider, in_features, out_features, scale, str(dtype))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if provider == "baseline1":
        mod = _load(INPUT_FILE, "baseline1")
    elif provider == "optimized":
        mod = _load(OPT_FILE, "optimized")
    else:
        return None
    torch.manual_seed(0)
    init = _init_args(batch, in_features, out_features, scale)
    if init == [()]:
        init = []
    m = mod.ModelNew(*init).eval().to("npu")
    _MODEL_CACHE[key] = m
    return m


def _run_torch_ref(x, batch, in_features, out_features, scale):
    m = _torch_model(batch, in_features, out_features, scale, x.dtype)
    y = F.linear(x.contiguous(), m.weight.to(x.dtype), m.bias.to(x.dtype))
    return F.silu(y) * scale


def _run_provider(provider, x, batch, in_features, out_features, scale):
    if provider == "torch":
        return _run_torch_ref(x, batch, in_features, out_features, scale)
    if provider == "baseline2":
        raise RuntimeError("baseline2_unavailable_reference_file_not_read_by_sandbox")
    m = _model(provider, batch, in_features, out_features, scale, x.dtype)
    return m(x)


def _sync():
    torch.npu.synchronize()


def _bench_call(fn, warmup=10, rep=30):
    if hasattr(triton.testing, "do_bench"):
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="mean")
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().detach().cpu())


def unit_test():
    ok_opt = True
    for label, batch, in_features, out_features in _BENCH_SHAPES:
        x = _make_input(batch, in_features)
        ref = _run_torch_ref(x, batch, in_features, out_features, 2.0)
        for provider in _PROVIDERS[1:]:
            display = {"baseline1": "Baseline Triton1", "baseline2": "Baseline Triton2", "optimized": "Optimized Triton"}[provider]
            if provider == "baseline2":
                print(f"TEST {display} {label}: SKIP_UNAVAILABLE reference_file_not_read max_abs=inf")
                continue
            try:
                y = _run_provider(provider, x, batch, in_features, out_features, 2.0)
                _sync()
                diff = _max_abs(y, ref)
                passed = diff <= 2e-2
                print(f"TEST {display} {label}: {'PASS' if passed else 'MISMATCH'} max_abs={diff:.6g}")
                if provider == "optimized" and not passed:
                    ok_opt = False
            except Exception as exc:
                print(f"TEST {display} {label}: EXCEPTION {type(exc).__name__} max_abs=inf")
                if provider == "optimized":
                    ok_opt = False
    # Forced persistent path for optimized kernel without huge allocation.
    try:
        opt = _load(OPT_FILE, "optimized")
        old = getattr(opt, "_MAX_PROGRAMS", None)
        opt._MAX_PROGRAMS = 1
        _MODEL_CACHE.clear()
        label, batch, in_features, out_features = ("forced_persistent_3x1024x8192", 3, 1024, 8192)
        x = _make_input(batch, in_features)
        ref = _run_torch_ref(x, batch, in_features, out_features, 2.0)
        y = _run_provider("optimized", x, batch, in_features, out_features, 2.0)
        _sync()
        diff = _max_abs(y, ref)
        passed = diff <= 2e-2
        print(f"TEST Optimized Triton {label}: {'PASS' if passed else 'MISMATCH'} max_abs={diff:.6g}")
        ok_opt = ok_opt and passed
    except Exception as exc:
        print(f"TEST Optimized Triton forced_persistent: EXCEPTION {type(exc).__name__} max_abs=inf")
        ok_opt = False
    finally:
        if 'opt' in locals() and old is not None:
            opt._MAX_PROGRAMS = old
        _MODEL_CACHE.clear()
    print("UNIT_TEST PASS" if ok_opt else "UNIT_TEST_FAILED")
    return ok_opt


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="matmul_swish_scaling_latency",
        args={},
    )
)
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, batch, in_features, out_features = shape
    if provider == "baseline2":
        print(f"INFO benchmark Baseline Triton2 {label}: inf reference_file_not_read")
        return float("inf")
    x = _make_input(batch, in_features)
    try:
        fn = lambda: _run_provider(provider, x, batch, in_features, out_features, 2.0)
        fn(); _sync()
        return _bench_call(fn)
    except Exception as exc:
        print(f"INFO benchmark {provider} {label}: inf {type(exc).__name__}")
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
