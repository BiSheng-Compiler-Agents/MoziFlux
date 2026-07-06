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
INPUT_FILE = ROOT / "63_Gemm_ReLU_Divide.py"
OPT_FILE = ROOT / "opt_63_Gemm_ReLU_Divide.py"
# The sandbox marks base_*.py as reference-only: keep the column visible but do not import it.
BASE2_PRESENT = (ROOT / "base_63_Gemm_ReLU_Divide.py").exists()

_BENCH_SHAPES = [
    ("small", 16, 128, 128),
    ("irregular", 33, 257, 513),
    ("target", 1024, 8192, 8192),
]
DIVISOR = 2.0
DTYPE = torch.float32
PROVIDERS = ["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"]
_MODULES = {}
_MODELS = {}
_REF_WEIGHTS = {}


def _load(path, name):
    if name in _MODULES:
        return _MODULES[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULES[name] = mod
    return mod


def _module_for(provider):
    if provider == "Baseline Triton1":
        return _load(INPUT_FILE, "k63_input_baseline")
    if provider == "Optimized Triton":
        return _load(OPT_FILE, "k63_optimized")
    return None


def _make_inputs(batch, in_features, out_features):
    torch.manual_seed(123)
    return (torch.rand(batch, in_features, device="npu", dtype=DTYPE),)


def _init_args(batch, in_features, out_features):
    return [in_features, out_features, DIVISOR]


def _model(provider, batch, in_features, out_features):
    key = (provider, in_features, out_features)
    if key in _MODELS:
        return _MODELS[key]
    mod = _module_for(provider)
    if mod is None:
        return None
    torch.manual_seed(0)
    model = mod.ModelNew(*_init_args(batch, in_features, out_features), dtype=DTYPE, device="npu")
    model.eval()
    _MODELS[key] = model
    return model


def _ref_weight_bias(in_features, out_features):
    key = (in_features, out_features)
    if key not in _REF_WEIGHTS:
        torch.manual_seed(0)
        linear = nn.Linear(in_features, out_features, bias=True, device="npu", dtype=DTYPE)
        linear.eval()
        _REF_WEIGHTS[key] = (linear.weight.detach(), linear.bias.detach())
    return _REF_WEIGHTS[key]


def _run_torch_ref(x, batch, in_features, out_features):
    weight, bias = _ref_weight_bias(in_features, out_features)
    y = F.linear(x, weight, bias)
    return torch.relu(y) / DIVISOR


def _run_provider(provider, x, batch, in_features, out_features):
    if provider == "PyTorch / ACL":
        return _run_torch_ref(x, batch, in_features, out_features)
    if provider == "Baseline Triton2":
        raise RuntimeError("reference_file_not_read_in_active_sandbox")
    model = _model(provider, batch, in_features, out_features)
    return model(x)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def _is_skip_provider(provider):
    return provider == "Baseline Triton2" and BASE2_PRESENT


def unit_test():
    ok = True
    for label, batch, in_features, out_features in _BENCH_SHAPES:
        x = _make_inputs(batch, in_features, out_features)[0]
        ref = _run_torch_ref(x, batch, in_features, out_features)
        for provider in PROVIDERS[1:]:
            if _is_skip_provider(provider):
                print(f"TEST {provider} {label}: SKIP_REFERENCE_FILE_NOT_READ max_abs=inf")
                continue
            try:
                out = _run_provider(provider, x, batch, in_features, out_features)
                torch.npu.synchronize()
                diff = _max_abs(out, ref)
                passed = diff <= 1e-3
                print(f"TEST {provider} {label}: {'PASS' if passed else 'MISMATCH'} max_abs={diff:.6g}")
                if provider == "Optimized Triton" and not passed:
                    ok = False
            except Exception as exc:
                etype = type(exc).__name__
                print(f"TEST {provider} {label}: SKIP_UNAVAILABLE {etype} max_abs=inf")
                if provider == "Optimized Triton":
                    ok = False

    # Force persistent fallback without allocating a huge tensor.
    try:
        opt = _load(OPT_FILE, "k63_optimized")
        old = opt._MAX_PROGRAMS
        opt._MAX_PROGRAMS = 1
        _MODELS.clear()
        batch, in_features, out_features = 3, 4096, 64
        x = _make_inputs(batch, in_features, out_features)[0]
        ref = _run_torch_ref(x, batch, in_features, out_features)
        out = _run_provider("Optimized Triton", x, batch, in_features, out_features)
        torch.npu.synchronize()
        diff = _max_abs(out, ref)
        passed = diff <= 1e-3
        print(f"TEST Optimized Triton forced_persistent: {'PASS' if passed else 'MISMATCH'} max_abs={diff:.6g}")
        ok = ok and passed
    except Exception as exc:
        print(f"TEST Optimized Triton forced_persistent: SKIP_UNAVAILABLE {type(exc).__name__} max_abs=inf")
        ok = False
    finally:
        try:
            opt._MAX_PROGRAMS = old
        except Exception:
            pass
        _MODELS.clear()

    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_ms(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / rep


def _bench_cell(provider, label, batch, in_features, out_features):
    if _is_skip_provider(provider):
        print(f"INFO bench_skip {provider} {label}: reference_file_not_read")
        return float("inf")
    x = _make_inputs(batch, in_features, out_features)[0]
    try:
        return _time_ms(lambda: _run_provider(provider, x, batch, in_features, out_features))
    except Exception as exc:
        print(f"INFO bench_unavailable {provider} {label}: {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDERS,
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="Latency (ms)",
        plot_name="gemm_relu_divide_latency",
        args={},
    )
)
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, batch, in_features, out_features = shape
    return _bench_cell(provider, label, batch, in_features, out_features)


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
        bench.run(print_data=True, show_plots=False, save_path=str(ROOT))


if __name__ == "__main__":
    main()
