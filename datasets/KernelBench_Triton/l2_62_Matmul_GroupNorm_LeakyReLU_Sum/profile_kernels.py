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

HERE = Path(__file__).resolve().parent
INPUT_FILE = "62_Matmul_GroupNorm_LeakyReLU_Sum.py"
OPT_FILE = "opt_62_Matmul_GroupNorm_LeakyReLU_Sum.py"
BASE_FILE = "base_62_Matmul_GroupNorm_LeakyReLU_Sum.py"  # intentionally not imported in this sandbox

_BENCH_SHAPES = [
    ("small", 128, 1024, 1024, 64),
    ("medium", 256, 2048, 2048, 128),
    ("irregular", 96, 1536, 1792, 112),
    ("target", 1024, 8192, 8192, 512),
]
_MODEL_CACHE = {}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(label, N, K, C, G):
    torch.manual_seed(123)
    return [torch.rand((N, K), device="npu", dtype=torch.float16)]


class TorchRef(nn.Module):

    def __init__(self, K, C, G, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(K, C)
        self.gn = nn.GroupNorm(num_groups=G, num_channels=C, eps=eps)
        self.negative_slope = negative_slope

    def forward(self, x):
        z = F.linear(x.contiguous(), self.fc.weight.to(x.device, x.dtype),
                     self.fc.bias.to(x.device, x.dtype))
        y = F.group_norm(z, self.gn.num_groups,
                         self.gn.weight.to(x.device, x.dtype),
                         self.gn.bias.to(x.device, x.dtype), self.gn.eps)
        return F.leaky_relu(y, negative_slope=self.negative_slope) * 2.0


def _model(key, K, C, G):
    ck = (key, K, C, G)
    if ck in _MODEL_CACHE:
        return _MODEL_CACHE[ck]
    torch.manual_seed(0)
    if key == "ref":
        m = TorchRef(K, C, G).eval().npu()
    elif key == "input":
        mod = _load(INPUT_FILE, "k_input_62")
        m = mod.ModelNew(K, C, G).eval().npu()
    elif key == "opt":
        mod = _load(OPT_FILE, "k_opt_62")
        m = mod.ModelNew(K, C, G).eval().npu()
    elif key == "opt_epilogue":
        mod = _load(OPT_FILE, "k_opt_62_ep")
        old = mod._USE_TRITON_EPILOGUE
        mod._USE_TRITON_EPILOGUE = True
        m = mod.ModelNew(K, C, G).eval().npu()
        mod._USE_TRITON_EPILOGUE = old
        m._force_epilogue_mod = mod
    else:
        raise KeyError(key)
    _MODEL_CACHE[ck] = m
    return m


def _run_torch_ref(x, K, C, G):
    return _model("ref", K, C, G)(x)


def _run_provider(key, x, K, C, G):
    if key == "base2":
        raise RuntimeError(
            "Baseline Triton2 unavailable: sandbox forbids reading base_*.py")
    if key == "opt_epilogue":
        m = _model("opt_epilogue", K, C, G)
        mod = m._force_epilogue_mod
        old = mod._USE_TRITON_EPILOGUE
        mod._USE_TRITON_EPILOGUE = True
        try:
            return m(x)
        finally:
            mod._USE_TRITON_EPILOGUE = old
    return _model(key, K, C, G)(x)


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().detach().cpu())


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label, N, K, C, G = shape
        x, = _make_inputs(*shape)
        ref = _run_torch_ref(x, K, C, G)
        for display, key in [("Baseline Triton1", "input"),
                             ("Baseline Triton2", "base2"),
                             ("Optimized Triton", "opt")]:
            try:
                y = _run_provider(key, x, K, C, G)
                diff = _max_abs(y, ref)
                passed = math.isfinite(diff) and diff <= 5e-2
                if display == "Optimized Triton":
                    ok = ok and passed
                status = "PASS" if passed else "MISMATCH"
                print(f"TEST {display} {label}: {status} max_abs={diff:.6g}")
            except Exception as e:
                if display == "Optimized Triton":
                    ok = False
                    print(
                        f"TEST {display} {label}: ERROR {type(e).__name__} max_abs=inf"
                    )
                else:
                    print(
                        f"TEST {display} {label}: SKIP_UNAVAILABLE {type(e).__name__} max_abs=inf"
                    )
        if label in ("small", "irregular"):
            try:
                y = _run_provider("opt_epilogue", x, K, C, G)
                diff = _max_abs(y, ref)
                passed = math.isfinite(diff) and diff <= 5e-2
                ok = ok and passed
                print(
                    f"TEST Optimized Triton forced_epilogue_{label}: {'PASS' if passed else 'MISMATCH'} max_abs={diff:.6g}"
                )
            except Exception as e:
                ok = False
                print(
                    f"TEST Optimized Triton forced_epilogue_{label}: ERROR {type(e).__name__} max_abs=inf"
                )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_one(provider, label):
    match = [s for s in _BENCH_SHAPES if s[0] == label][0]
    _, N, K, C, G = match
    x, = _make_inputs(*match)
    if provider == "torch":

        def fn():
            return _run_torch_ref(x, K, C, G)
    elif provider == "input":

        def fn():
            return _run_provider("input", x, K, C, G)
    elif provider == "base2":
        print(
            f"INFO benchmark_preskip Baseline Triton2 {label}: sandbox_base_file_not_read"
        )
        return float("inf")
    elif provider == "opt":

        def fn():
            return _run_provider("opt", x, K, C, G)
    else:
        return float("inf")
    try:
        for _ in range(5):
            fn()
            torch.npu.synchronize()
        if hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn,
                                           warmup=10,
                                           rep=30,
                                           return_mode="mean")
        start = time.perf_counter()
        for _ in range(30):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - start) * 1000.0 / 30.0
    except Exception as e:
        print(
            f"INFO benchmark_unavailable {provider} {label}: {type(e).__name__}"
        )
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "input", "base2", "opt"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="matmul_groupnorm_lrelu_sum",
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
