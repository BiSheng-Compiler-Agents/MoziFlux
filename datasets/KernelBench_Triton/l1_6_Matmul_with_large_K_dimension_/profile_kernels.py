import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

ROOT = Path(__file__).resolve().parent
PROVIDERS = {
    "torch": ("PyTorch / ACL", None),
    "baseline1": ("Baseline Triton1", "6_Matmul_with_large_K_dimension_.py"),
    "baseline2":
    ("Baseline Triton2", "base_6_Matmul_with_large_K_dimension_.py"),
    "optimized":
    ("Optimized Triton", "opt_6_Matmul_with_large_K_dimension_.py"),
}
_BENCH_SHAPES = {
    "direct_small": (64, 64, 131072 * 4),
    "direct_nonpow2": (129, 96, 131072 * 4),
    "split_medium": (128, 128, 131072 * 4),
    "benchmark_largeK": (256, 256, 131072 * 4),
}

_MODEL_CACHE = {}


def _load(path):
    name = "k_" + path.replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _model(key):
    if key not in _MODEL_CACHE:
        _, file_name = PROVIDERS[key]
        mod = _load(file_name)
        _MODEL_CACHE[key] = mod.ModelNew()
    return _MODEL_CACHE[key]


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _make_inputs(shape, seed=0):
    M, N, K = shape
    torch.manual_seed(seed)
    A = torch.randn((M, K), device="npu", dtype=torch.float32) * 0.01
    B = torch.randn((K, N), device="npu", dtype=torch.float32) * 0.01
    return A, B


def _run_torch_ref(A, B):
    return torch.matmul(A, B)


def _run_provider(key, A, B):
    if key == "torch":
        return _run_torch_ref(A, B)
    return _model(key)(A, B)


def _bench_one(fn, warmup=10, rep=50):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
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
        x_vals=list(_BENCH_SHAPES.keys()),
        line_arg="provider",
        line_vals=["torch", "baseline1", "baseline2", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="matmul_large_k",
        args={},
    ))
def benchmark(label, provider):
    try:
        A, B = _make_inputs(_BENCH_SHAPES[label], seed=123)

        def fn():
            return _run_provider(provider, A, B)

        return _bench_one(fn, warmup=25, rep=100)
    except Exception as exc:
        print(
            f"ERROR provider={provider} label={label}: {type(exc).__name__}: {exc}"
        )
        return float("inf")


def unit_test():
    ok = True
    for label, shape in _BENCH_SHAPES.items():
        A, B = _make_inputs(shape, seed=7)
        ref = _run_torch_ref(A, B)
        _sync()
        for key in ("baseline1", "baseline2", "optimized"):
            try:
                out = _run_provider(key, A, B)
                _sync()
                close = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
                max_abs = (out - ref).abs().max().item()
                print(
                    f"TEST {key:9s} {label:16s}: {'PASS' if close else 'FAIL'} max_abs={max_abs:.6g}"
                )
                ok = ok and bool(close)
            except Exception as exc:
                ok = False
                print(
                    f"TEST {key:9s} {label:16s}: ERROR {type(exc).__name__}: {exc}"
                )
    return ok


def run_bench():
    txt = benchmark.run(print_data=True, show_plots=False, save_path=str(ROOT))
    return txt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not args.test and not args.bench:
        args.test = args.bench = True
    ok = True
    if args.test:
        ok = unit_test()
    if args.bench:
        run_bench()
    # Keep exit code zero so benchmark artifacts/results.txt are preserved for parsers.
    if not ok:
        print("UNIT_TEST_FAILED")


if __name__ == "__main__":
    main()
