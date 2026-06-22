import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import triton

ROOT = Path(__file__).resolve().parent
_MODEL_CACHE = {}

PROVIDERS = [
    ("torch", "PyTorch / ACL", None),
    ("baseline1", "Baseline Triton1", "35_GroupNorm_.py"),
    ("baseline2", "Baseline Triton2", "base_35_GroupNorm_.py"),
    ("optimized", "Optimized Triton", "opt_35_GroupNorm_.py"),
]

_BENCH_SHAPES = [
    ("small_direct", 2, 64, 16, 16),
    ("medium_direct", 8, 64, 64, 64),
    ("benchmark_persistent", 112, 64, 512, 512),
]


def _load(key, filename):
    if key in _MODEL_CACHE and key.endswith("_mod"):
        return _MODEL_CACHE[key]
    path = ROOT / filename
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"k35_{key}_{path.stem}",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODEL_CACHE[key + "_mod"] = mod
    return mod


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if key == "torch":
        m = nn.GroupNorm(num_groups=8, num_channels=64).to(device="npu").eval()
    else:
        file = dict((k, f) for k, _, f in PROVIDERS)[key]
        mod = _load(key, file)
        if mod is None:
            return None
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        if init == [()]:
            init = []
        m = mod.ModelNew(*init).to(device="npu").eval()
    _MODEL_CACHE[key] = m
    return m


def _make_inputs(shape):
    label, n, c, h, w = shape
    torch.manual_seed(0)
    x = torch.rand((n, c, h, w), device="npu", dtype=torch.float32)
    return [x]


def _run_provider(key, x):
    m = _model(key)
    if m is None:
        raise RuntimeError(f"provider {key} missing")
    with torch.no_grad():
        return m(x)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _close(a, b):
    diff = (a - b).abs()
    max_abs = float(diff.max().detach().cpu())
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-6)
    max_rel = float((diff / denom).max().detach().cpu())
    ok = torch.allclose(a, b, rtol=1e-3, atol=1e-3)
    return bool(ok), max_abs, max_rel


def unit_test():
    optimized_ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_inputs(shape)[0]
        ref = _run_provider("torch", x)
        _sync()
        for key, name, _ in PROVIDERS[1:]:
            try:
                y = _run_provider(key, x)
                _sync()
                ok, max_abs, max_rel = _close(y, ref)
                status = "PASS" if ok else "MISMATCH"
                print(
                    f"TEST {key} {label} {status} max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
                )
                if key == "optimized" and not ok:
                    optimized_ok = False
            except Exception as e:
                print(
                    f"TEST {key} {label} INFO_EXCEPTION {type(e).__name__}: {e}"
                )
                if key == "optimized":
                    optimized_ok = False
    persistent_shape = ("dispatch_persistent", 1025, 64, 1, 1)
    x = _make_inputs(persistent_shape)[0]
    ref = _run_provider("torch", x)
    try:
        y = _run_provider("optimized", x)
        _sync()
        ok, max_abs, max_rel = _close(y, ref)
        status = "PASS" if ok else "MISMATCH"
        print(
            f"TEST optimized dispatch_persistent {status} max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
        )
        optimized_ok = optimized_ok and ok
    except Exception as e:
        print(
            f"TEST optimized dispatch_persistent INFO_EXCEPTION {type(e).__name__}: {e}"
        )
        optimized_ok = False
    print("UNIT_TEST PASS" if optimized_ok else "UNIT_TEST_FAILED")
    return optimized_ok


def _manual_bench(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
        _sync()
    torch.cuda.Event(enable_timing=True) if False else None
    import time
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
        _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


def _bench_one(provider, label):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = _make_inputs(shape)[0]
    try:

        def fn():
            return _run_provider(provider, x)

        fn()
        _sync()
        if hasattr(triton, "testing") and hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn,
                                           warmup=10,
                                           rep=50,
                                           return_mode="mean")
        return _manual_bench(fn)
    except Exception as e:
        print(
            f"INFO bench {provider} {label} exception {type(e).__name__}: {e}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "baseline1", "baseline2", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="groupnorm-performance",
        args={},
    ))
def bench(label, provider):
    return _bench_one(provider, label)


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


if __name__ == "__main__":
    main()
