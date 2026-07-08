import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import triton

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "39_Gemm_Scale_BatchNorm.py"
BASE_FILE = HERE / "base_39_Gemm_Scale_BatchNorm.py"
OPT_FILE = HERE / "opt_39_Gemm_Scale_BatchNorm.py"

_BENCH_SHAPES = [
    ("small_M128_K1024_N512", 128, 1024, 512),
    ("medium_M1024_K2048_N1024", 1024, 2048, 1024),
    ("irregular_M257_K1024_N768", 257, 1024, 768),
    ("required_M16384_K4096_N4096", 16384, 4096, 4096),
]
_SHAPE_BY_LABEL = {s[0]: s[1:] for s in _BENCH_SHAPES}
_PROVIDER_KEYS = ["torch", "baseline1", "baseline2", "opt"]
_MODEL_CACHE = {}
_MODULE_CACHE = {}
_BASE2_IMPORT_ERROR = None


def _load(path: Path, name: str):
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[name] = mod
    return mod


class TorchRef(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 scale_shape,
                 eps=1e-5,
                 momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x_fp32 = x if x.dtype == torch.float32 else x.to(torch.float32)
        y = torch.nn.functional.linear(x_fp32, self.gemm.weight,
                                       self.gemm.bias)
        y = y * self.scale
        return self.bn(y)


def _model(provider, K, N):
    global _BASE2_IMPORT_ERROR
    key = (provider, K, N)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    if provider == "torch":
        model = TorchRef(K, N, (N, ))
    elif provider == "baseline1":
        mod = _load(INPUT_FILE, "k_l2_39_input")
        model = mod.ModelNew(K, N, (N, ))
    elif provider == "baseline2":
        try:
            mod = _load(BASE_FILE, "k_l2_39_base")
            model = mod.ModelNew(K, N, (N, ))
        except Exception as exc:
            _BASE2_IMPORT_ERROR = type(exc).__name__
            return None
    elif provider == "opt":
        mod = _load(OPT_FILE, "k_l2_39_opt")
        model = mod.ModelNew(K, N, (N, ))
    else:
        raise KeyError(provider)
    model.eval().to("npu")
    _MODEL_CACHE[key] = model
    return model


def _make_inputs(M, K, dtype=torch.float32):
    torch.manual_seed(123)
    return torch.rand((M, K), device="npu", dtype=dtype)


def _run_provider(provider, x, K, N):
    model = _model(provider, K, N)
    if model is None:
        raise RuntimeError("comparison provider unavailable")
    with torch.no_grad():
        return model(x)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _bench_callable(fn):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=25,
                                       rep=100,
                                       return_mode="mean")
    except Exception:
        import time
        for _ in range(10):
            fn()
            _sync()
        t0 = time.perf_counter()
        reps = 50
        for _ in range(reps):
            fn()
        _sync()
        return (time.perf_counter() - t0) * 1000.0 / reps


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDER_KEYS,
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("green", "-"), ("black", "--"), ("red", "-")],
        ylabel="Latency (ms)",
        plot_name="gemm_scale_batchnorm",
        args={},
    ))
def benchmark(label, provider):
    M, K, N = _SHAPE_BY_LABEL[label]
    if provider == "baseline2" and _BASE2_IMPORT_ERROR is not None:
        print(f"INFO baseline2_unavailable {label}: {_BASE2_IMPORT_ERROR}")
        return float("inf")
    x = _make_inputs(M, K)
    try:
        _run_provider(provider, x, K, N)
        _sync()
        return _bench_callable(lambda: _run_provider(provider, x, K, N))
    except Exception as exc:
        print(
            f"INFO provider_timing_unavailable {provider} {label}: {type(exc).__name__}"
        )
        return float("inf")


def unit_test():
    ok = True
    for label, M, K, N in _BENCH_SHAPES:
        x = _make_inputs(M, K)
        ref = _run_provider("torch", x, K, N)
        _sync()
        for provider, name in [("baseline1", "Baseline Triton1"),
                               ("baseline2", "Baseline Triton2"),
                               ("opt", "Optimized Triton")]:
            try:
                out = _run_provider(provider, x, K, N)
                _sync()
                torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
                print(f"PASS {name} {label}")
            except Exception as exc:
                if provider == "opt":
                    ok = False
                    print(
                        f"UNIT_TEST_FAILED {name} {label} {type(exc).__name__}"
                    )
                else:
                    print(
                        f"INFO {name} {label} comparison_unavailable_or_mismatch {type(exc).__name__}"
                    )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


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
        benchmark.run(print_data=True, show_plots=False, save_path=str(HERE))


if __name__ == "__main__":
    main()
