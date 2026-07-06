import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton

HERE = Path(__file__).resolve().parent
INPUT_FILE = "45_Gemm_Sigmoid_Sum_LogSumExp.py"
BASE_FILE = "base_45_Gemm_Sigmoid_Sum_LogSumExp.py"
OPT_FILE = "opt_45_Gemm_Sigmoid_Sum_LogSumExp.py"

_BENCH_SHAPES = [
    ("default_B128_K10_H20", 128, 10, 20, 5),
    ("boundary_B17_K10_H20", 17, 10, 20, 5),
    ("aligned_B64_K16_H32", 64, 16, 32, 5),
]

_MOD_CACHE = {}
_MODEL_CACHE = {}


def _load(filename, key):
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(f"k_l2_45_{key}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MOD_CACHE[key] = mod
    return mod


def _make_inputs(B, K, H, O):
    torch.manual_seed(123)
    return (torch.randn((B, K), device="npu", dtype=torch.float32),)


class TorchRef(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = torch.sigmoid(self.linear1(x))
        return torch.logsumexp(y.sum(dim=1), dim=0)


def _model(provider, K, H, O):
    cache_key = (provider, K, H, O)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    torch.manual_seed(0)
    if provider == "PyTorch / ACL":
        model = TorchRef(K, H, O).npu().eval()
    else:
        fname = {"Baseline Triton1": INPUT_FILE, "Baseline Triton2": BASE_FILE, "Optimized Triton": OPT_FILE}[provider]
        mod = _load(fname, provider.replace(" ", "_").replace("/", "_"))
        model = mod.ModelNew(K, H, O).npu().eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _run_provider(provider, x, B, K, H, O):
    if provider == "Baseline Triton2" and not (HERE / BASE_FILE).exists():
        raise RuntimeError("base file missing")
    model = _model(provider, K, H, O)
    with torch.no_grad():
        return model(x)


def _run_torch_ref(x, B, K, H, O):
    return _run_provider("PyTorch / ACL", x, B, K, H, O)


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().detach().cpu())


def unit_test():
    ok_opt = True
    providers = ["Baseline Triton1", "Baseline Triton2", "Optimized Triton"]
    for label, B, K, H, O in _BENCH_SHAPES:
        x, = _make_inputs(B, K, H, O)
        ref = _run_torch_ref(x, B, K, H, O)
        for provider in providers:
            try:
                out = _run_provider(provider, x, B, K, H, O)
                diff = _max_abs(out, ref)
                passed = math.isfinite(diff) and diff <= 1e-3
                print(f"CHECK {provider} {label}: max_abs={diff:.6g} status={'PASS' if passed else 'MISMATCH'}")
                if provider == "Optimized Triton" and not passed:
                    ok_opt = False
            except Exception as exc:
                print(f"INFO provider_unavailable {provider} {label}: {type(exc).__name__}")
                if provider == "Optimized Triton":
                    ok_opt = False
    print("UNIT_TEST PASS" if ok_opt else "UNIT_TEST_FAILED")
    return ok_opt


def _bench_one(provider, label):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, B, K, H, O = shape
    x, = _make_inputs(B, K, H, O)
    try:
        # Correctness gate for this cell before timing.
        ref = _run_torch_ref(x, B, K, H, O)
        out = _run_provider(provider, x, B, K, H, O)
        if _max_abs(out, ref) > 1e-3:
            print(f"INFO benchmark_preskip_mismatch {provider} {label}")
            return float("inf")
        torch.npu.synchronize()
        fn = lambda: _run_provider(provider, x, B, K, H, O)
        try:
            return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")
        except Exception:
            for _ in range(10):
                fn()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                fn()
            torch.npu.synchronize()
            return (time.perf_counter() - t0) * 1000.0 / 100.0
    except Exception as exc:
        print(f"INFO benchmark_unavailable {provider} {label}: {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="l2_45_gemm_sigmoid_sum_logsumexp",
        args={},
    )
)
def benchmark(label, provider):
    return _bench_one(provider, label)


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
        benchmark.run(print_data=True, show_plots=False, save_path=None)


if __name__ == "__main__":
    main()
