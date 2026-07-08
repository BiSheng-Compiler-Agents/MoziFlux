import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "76_Gemm_Add_ReLU.py"
OPT_FILE = ROOT / "opt_76_Gemm_Add_ReLU.py"
# base_*.py is intentionally not imported: the active sandbox marks reference files as DO NOT read.

_BENCH_SHAPES = [
    ("small_acl_fallback", 128, 512, 512),
    ("large_triton_irregular", 384, 1536, 2048),
    ("target_1024x8192x8192", 1024, 8192, 8192),
]

_PROVIDERS = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}
_MODULE_CACHE = {}
_MODEL_CACHE = {}


class TorchRef(nn.Module):

    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        w = self.gemm.weight.detach().to(device=x.device, dtype=x.dtype)
        b = self.bias.detach().to(device=x.device, dtype=x.dtype)
        return torch.relu(torch.matmul(x, w.transpose(0, 1)) + b)


def _load(path: Path, key: str):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"k_l2_76_{key}_{path.stem}",
                                                  path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def _init_inputs(in_features, out_features):
    return [in_features, out_features, (out_features, )]


def _model(key: str, in_features: int, out_features: int):
    cache_key = (key, in_features, out_features)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    torch.manual_seed(0)
    if key == "torch":
        model = TorchRef(*_init_inputs(in_features, out_features))
    elif key == "baseline1":
        model = _load(
            INPUT_FILE,
            "baseline1").ModelNew(*_init_inputs(in_features, out_features))
    elif key == "optimized":
        model = _load(
            OPT_FILE,
            "optimized").ModelNew(*_init_inputs(in_features, out_features))
    else:
        raise RuntimeError("baseline2_unavailable_reference_file_not_read")
    model.eval().npu()
    _MODEL_CACHE[cache_key] = model
    return model


def _make_input(m: int, k: int):
    torch.manual_seed(123)
    return torch.rand((m, k), device="npu", dtype=torch.float16)


def _run_provider(provider: str, label: str):
    _, m, k, n = _shape(label)
    x = _make_input(m, k)
    if provider == "baseline2":
        raise RuntimeError("baseline2_unavailable_reference_file_not_read")
    with torch.no_grad():
        return _model(provider, k, n)(x)


def _shape(label: str):
    for row in _BENCH_SHAPES:
        if row[0] == label:
            return row
    raise KeyError(label)


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().detach().cpu().item())


def unit_test():
    ok = True
    for label, m, k, n in _BENCH_SHAPES:
        try:
            ref = _run_provider("torch", label)
            print(f"TEST PyTorch / ACL {label}: PASS max_abs=0.0")
        except Exception as exc:
            print(
                f"TEST PyTorch / ACL {label}: FAIL {type(exc).__name__} max_abs=inf"
            )
            ok = False
            continue

        print(
            f"TEST Baseline Triton2 {label}: SKIP_UNAVAILABLE SandboxReferenceNotRead max_abs=inf"
        )
        for provider in ("baseline1", "optimized"):
            name = _PROVIDERS[provider]
            try:
                out = _run_provider(provider, label)
                torch.npu.synchronize()
                diff = _max_abs(out, ref)
                passed = math.isfinite(diff) and diff <= 1e-2
                print(
                    f"TEST {name} {label}: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}"
                )
                if provider == "optimized" and not passed:
                    ok = False
            except Exception as exc:
                # Comparison provider failures stay visible; optimized failures gate UNIT_TEST.
                status = "FAIL" if provider == "optimized" else "SKIP_UNAVAILABLE"
                print(
                    f"TEST {name} {label}: {status} {type(exc).__name__} max_abs=inf"
                )
                if provider == "optimized":
                    ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_ms(fn, warmup=3, rep=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / rep


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[row[0] for row in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "baseline1", "baseline2", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="l2_76_gemm_add_relu",
        args={},
    ))
def bench(label, provider):
    if provider == "baseline2":
        return float("inf")
    try:

        def fn():
            return _run_provider(provider, label)

        try:
            return triton.testing.do_bench(fn,
                                           warmup=5,
                                           rep=20,
                                           return_mode="mean")
        except Exception:
            return _time_ms(fn, warmup=1, rep=3)
    except Exception as exc:
        print(
            f"INFO benchmark_unavailable {_PROVIDERS[provider]} {label}: {type(exc).__name__}"
        )
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
        bench.run(print_data=True,
                  show_plots=False,
                  save_path=str(ROOT / "profile_plots"))


if __name__ == "__main__":
    main()
