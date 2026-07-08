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
INPUT_FILE = ROOT / "64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py"
OPT_FILE = ROOT / "opt_64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py"
# The sandbox explicitly marks base_*.py as DO NOT read; keep Baseline Triton2 visible but skipped.
BASE_FILE = ROOT / "base_64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py"

_BENCH_SHAPES = [
    ("small", 128, 1024, 512),
    ("medium", 512, 4096, 2048),
    ("target", 1024, 8192, 8192),
]
_SHAPE_BY_LABEL = {s[0]: s for s in _BENCH_SHAPES}
_LINE_NAMES = [
    "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
]
_MODEL_CACHE = {}
_MODULE_CACHE = {}


def _load(path, key):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _device():
    return torch.device("npu") if hasattr(torch,
                                          "npu") else torch.device("cuda")


def _make_inputs(label):
    _, B, IN_FEATURES, OUT_FEATURES = _SHAPE_BY_LABEL[label]
    torch.manual_seed(123)
    return (torch.rand(B, IN_FEATURES,
                       device=_device()), IN_FEATURES, OUT_FEATURES)


def _init_linear(in_features, out_features):
    torch.manual_seed(0)
    return nn.Linear(in_features, out_features, bias=True).to(_device())


def _torch_ref(x, in_features, out_features):
    linear = _init_linear(in_features, out_features)
    y = linear(x)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = F.leaky_relu(y, negative_slope=0.01)
    y = F.leaky_relu(y, negative_slope=0.01)
    y = F.gelu(y)
    y = F.gelu(y)
    return y


def _model(provider, in_features, out_features):
    key = (provider, in_features, out_features)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if provider == "baseline1":
        mod = _load(INPUT_FILE, "baseline1")
    elif provider == "opt":
        mod = _load(OPT_FILE, "opt")
    else:
        raise KeyError(provider)
    torch.manual_seed(0)
    m = mod.ModelNew(in_features, out_features).to(_device())
    _MODEL_CACHE[key] = m
    return m


def _run_provider(provider, x, in_features, out_features):
    if provider == "torch":
        return _torch_ref(x, in_features, out_features)
    if provider == "baseline2":
        raise RuntimeError("baseline2_skipped_by_sandbox_do_not_read")
    return _model(provider, in_features, out_features)(x)


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def unit_test():
    ok = True
    for label, B, IN_FEATURES, OUT_FEATURES in _BENCH_SHAPES:
        x, in_features, out_features = _make_inputs(label)
        ref = _torch_ref(x, in_features, out_features)
        for display, provider in [("Baseline Triton1", "baseline1"),
                                  ("Baseline Triton2", "baseline2"),
                                  ("Optimized Triton", "opt")]:
            if provider == "baseline2":
                print(
                    f"TEST {display} {label}: SKIP_UNAVAILABLE sandbox_do_not_read max_abs=inf"
                )
                continue
            try:
                out = _run_provider(provider, x, in_features, out_features)
                _sync()
                diff = _max_abs(out, ref)
                passed = math.isfinite(diff) and diff <= 1e-3
                print(
                    f"TEST {display} {label}: {'PASS' if passed else 'MISMATCH'} max_abs={diff:.6g}"
                )
                if provider == "opt" and not passed:
                    ok = False
            except Exception as e:
                print(
                    f"TEST {display} {label}: SKIP_UNAVAILABLE {type(e).__name__} max_abs=inf"
                )
                if provider == "opt":
                    ok = False
        # Force-test optimized Triton fallback paths on a modest shape without changing production routing.
        if label == "small":
            try:
                opt = _load(OPT_FILE, "opt")
                m = _model("opt", in_features, out_features)
                z = m.linear(x).contiguous()
                fallback_direct = opt._post_lse_triton(z,
                                                       m.neg_slope,
                                                       force_persistent=False)
                fallback_persistent = opt._post_lse_triton(
                    z, m.neg_slope, force_persistent=True)
                acl = opt._post_lse_acl(z, m.neg_slope)
                _sync()
                d1 = _max_abs(fallback_direct, acl)
                d2 = _max_abs(fallback_persistent, acl)
                p1 = d1 <= 1e-3
                p2 = d2 <= 1e-3
                print(
                    f"TEST Optimized Triton fallback_direct {label}: {'PASS' if p1 else 'MISMATCH'} max_abs={d1:.6g}"
                )
                print(
                    f"TEST Optimized Triton fallback_persistent {label}: {'PASS' if p2 else 'MISMATCH'} max_abs={d2:.6g}"
                )
                ok = ok and p1 and p2
            except Exception as e:
                print(
                    f"TEST Optimized Triton fallback_paths {label}: SKIP_UNAVAILABLE {type(e).__name__} max_abs=inf"
                )
                ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_ms(fn, warmup=5, rep=20):
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
        line_vals=_LINE_NAMES,
        line_names=_LINE_NAMES,
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="gemm_lse_leaky_gelu2",
        args={},
    ))
def bench(label, provider):
    x, in_features, out_features = _make_inputs(label)
    if provider == "Baseline Triton2":
        print(f"INFO benchmark_preskip {provider} {label} sandbox_do_not_read")
        return float("inf")
    key = {
        "PyTorch / ACL": "torch",
        "Baseline Triton1": "baseline1",
        "Optimized Triton": "opt"
    }[provider]
    try:
        reps = 5 if label == "target" else 20
        warms = 2 if label == "target" else 5
        return _time_ms(
            lambda: _run_provider(key, x, in_features, out_features),
            warmup=warms,
            rep=reps)
    except Exception as e:
        print(
            f"INFO benchmark_unavailable {provider} {label} {type(e).__name__}"
        )
        return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if not args.test and not args.bench:
        args.test = args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        bench.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
