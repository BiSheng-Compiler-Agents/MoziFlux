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
INPUT_FILE = ROOT / "56_Matmul_Sigmoid_Sum.py"
BASE_FILE = ROOT / "base_56_Matmul_Sigmoid_Sum.py"
OPT_FILE = ROOT / "opt_56_Matmul_Sigmoid_Sum.py"

_BENCH_SHAPES = [
    ("tiny_irregular", 3, 257, 193),
    ("small_aligned", 16, 1024, 1024),
    ("medium_largeK", 32, 4096, 4096),
    ("default", 128, 32768, 32768),
]

_MODULE_CACHE = {}
_MODEL_CACHE = {}


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _load(path: Path, key: str):
    cache_key = (key, str(path))
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[cache_key] = mod
    return mod


def _model(provider: str, I: int, H: int):
    key = (provider, I, H)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    if provider == "torch":
        model = nn.Linear(I, H).npu().eval()
    else:
        path = {"baseline1": INPUT_FILE, "baseline2": BASE_FILE, "opt": OPT_FILE}[provider]
        mod = _load(path, provider)
        model = mod.ModelNew(I, H).npu().eval()
    _MODEL_CACHE[key] = model
    return model


def _make_inputs(B: int, I: int):
    torch.manual_seed(123)
    return torch.rand((B, I), device="npu", dtype=torch.float32)


def _run_torch_ref(x, I: int, H: int):
    model = _model("torch", I, H)
    return torch.sigmoid(F.linear(x, model.weight, model.bias)).sum(dim=1, keepdim=True)


def _run_provider(provider: str, x, I: int, H: int):
    if provider == "torch":
        return _run_torch_ref(x, I, H)
    model = _model(provider, I, H)
    return model(x)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok = True
    providers = ["baseline1", "baseline2", "opt"]
    for label, B, I, H in _BENCH_SHAPES:
        x = _make_inputs(B, I)
        ref = _run_torch_ref(x, I, H)
        _sync()
        for provider in providers:
            try:
                y = _run_provider(provider, x, I, H)
                _sync()
                diff = _max_abs(y.float(), ref.float())
                tol = 5e-2 if provider == "opt" else 1e-2
                if not math.isfinite(diff) or diff > tol:
                    print(f"UNIT {provider} {label} MISMATCH max_abs={diff:.6g} tol={tol}")
                    if provider == "opt":
                        ok = False
                else:
                    print(f"UNIT {provider} {label} PASS max_abs={diff:.6g}")
            except BaseException as exc:
                safe = type(exc).__name__
                if provider == "baseline2":
                    print(f"INFO baseline2 {label} unavailable_or_preskipped {safe}")
                else:
                    print(f"UNIT {provider} {label} EXCEPTION {safe}")
                    if provider == "opt":
                        ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_once(fn, warmup=5, rep=20):
    try:
        import triton.testing as tt
        return tt.do_bench(fn, warmup=warmup, rep=rep, return_mode="mean")
    except Exception:
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
        line_vals=["torch", "baseline1", "baseline2", "opt"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="matmul_sigmoid_sum_latency",
        args={},
    )
)
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, B, I, H = shape
    if provider == "baseline2" and not BASE_FILE.exists():
        print(f"INFO baseline2 {label} missing")
        return float("inf")
    try:
        x = _make_inputs(B, I)
        # Compile/warm model outside measured lambda when possible.
        _run_provider(provider, x, I, H)
        _sync()
        return _bench_once(lambda: _run_provider(provider, x, I, H))
    except BaseException as exc:
        print(f"INFO {provider} {label} benchmark_unavailable {type(exc).__name__}")
        return float("inf")


def run_bench():
    benchmark.run(print_data=True, show_plots=False, save_path=str(ROOT / "remote_results"))


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
        run_bench()


if __name__ == "__main__":
    main()
