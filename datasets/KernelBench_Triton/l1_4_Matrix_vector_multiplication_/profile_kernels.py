import argparse
import importlib.util
import time
from pathlib import Path

import torch
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
BASELINE_FILE1 = ROOT / "4_Matrix_vector_multiplication_.py"
BASELINE_FILE2 = ROOT / "base_4_Matrix_vector_multiplication_.py"
OPT_FILE = ROOT / "opt_4_Matrix_vector_multiplication_.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_MODULES = {}
_MODELS = {}
_BENCH_SHAPES = {
    "small": (64, 512),
    "nonpow2": (257, 777),
    "medium": (512, 8192),
    "benchmark": (2048, 1048576),
}


def _module(key: str):
    if key not in _MODULES:
        if key == "baseline1":
            _MODULES[key] = _load(BASELINE_FILE1, "k_l1_4_baseline1")
        elif key == "baseline2":
            _MODULES[key] = _load(BASELINE_FILE2, "k_l1_4_baseline2")
        elif key == "optimized":
            _MODULES[key] = _load(OPT_FILE, "k_l1_4_optimized")
        else:
            raise KeyError(key)
    return _MODULES[key]


def _model(key: str):
    if key not in _MODELS:
        _MODELS[key] = _module(key).ModelNew().to("npu")
    return _MODELS[key]


def _make_inputs(M: int, K: int, dtype=torch.float16):
    torch.manual_seed(0)
    A = torch.randn((M, K), device="npu", dtype=dtype)
    B = torch.randn((K, 1), device="npu", dtype=dtype)
    return A, B


def _run_torch_ref(A, B):
    return torch.matmul(A, B)


def _run_provider(provider: str, A, B):
    if provider == "torch":
        return _run_torch_ref(A, B)
    return _model(provider)(A, B)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _bench_callable(fn, warmup=25, rep=100):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(10):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        _sync()
        return (time.perf_counter() - t0) * 1000.0 / rep


def unit_test():
    cases = [
        (1, 1, torch.float32),
        (7, 31, torch.float16),
        (64, 512, torch.float16),
        (257, 777, torch.float16),
    ]
    ok = True
    for M, K, dtype in cases:
        A, B = _make_inputs(M, K, dtype)
        ref = _run_torch_ref(A, B)
        for provider in ("baseline1", "baseline2", "optimized"):
            try:
                out = _run_provider(provider, A, B)
                _sync()
                atol = 1e-2 if dtype is torch.float16 else 1e-3
                rtol = 1e-2 if dtype is torch.float16 else 1e-3
                passed = torch.allclose(out, ref, atol=atol, rtol=rtol)
                max_err = (out - ref).abs().max().item()
                print(
                    f"TEST {provider} M={M} K={K} dtype={dtype}: {'PASS' if passed else 'FAIL'} max_abs={max_err:.6g}"
                )
                ok = ok and bool(passed)
            except Exception as exc:
                print(
                    f"TEST {provider} M={M} K={K} dtype={dtype}: ERROR {exc}")
                ok = False
    print(f"UNIT_TEST {'PASS' if ok else 'FAIL'}")
    return ok


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
        styles=[("black", "-"), ("blue", "--"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="l1_4_gemv",
        args={},
    ))
def benchmark(label, provider):
    M, K = _BENCH_SHAPES[label]
    A, B = _make_inputs(M, K, torch.float16)
    try:
        _run_provider(provider, A, B)
        _sync()
        return _bench_callable(lambda: _run_provider(provider, A, B))
    except Exception as exc:
        print(f"BENCH_ERROR provider={provider} label={label}: {exc}")
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
