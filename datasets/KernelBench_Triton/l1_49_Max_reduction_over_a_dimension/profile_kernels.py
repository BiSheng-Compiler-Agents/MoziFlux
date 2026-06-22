import argparse
import importlib.util
import pathlib
import sys
import time

import torch
import triton

ROOT = pathlib.Path(__file__).resolve().parent
_FILES = {
    "baseline1": "49_Max_reduction_over_a_dimension.py",
    "baseline2": "base_49_Max_reduction_over_a_dimension.py",
    "optimized": "opt_49_Max_reduction_over_a_dimension.py",
}
_PROVIDER_NAMES = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}
_MODULES = {}
_MODEL_CACHE = {}

# labels contain no spaces; target_original preserves get_inputs() shape and default dim=1.
_BENCH_SHAPES = [
    ("small_dim1", 4, 257, 255, 1),
    ("medium_dim1", 16, 1024, 1023, 1),
    ("target_original", 128, 4096, 4095, 1),
]
_TEST_SHAPES = [
    ("dim0_path", 5, 17, 33, 0),
    ("dim1_path", 4, 257, 255, 1),
    ("dim2_path", 3, 19, 257, 2),
    ("neg_dim1", 4, 129, 127, -2),
]


def _load(key):
    if key in _MODULES:
        return _MODULES[key]
    path = ROOT / _FILES[key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULES[key] = mod
    return mod


def _model(key, dim):
    cache_key = (key, int(dim))
    if cache_key not in _MODEL_CACHE:
        mod = _load(key)
        _MODEL_CACHE[cache_key] = mod.ModelNew(
            int(dim)).to(device="npu").eval()
    return _MODEL_CACHE[cache_key]


def _make_inputs(B, M, N, dtype=torch.float32):
    # Match source get_inputs(): torch.rand(...), contiguous 3D tensor.
    return torch.rand((B, M, N), device="npu", dtype=dtype)


def _run_torch_ref(x, dim):
    return torch.max(x, dim=dim).values


def _baseline_grid_overflows(key, B, M, N, dim):
    # Pre-skip only known original-style 2D grids that would exceed Ascend coreDim.
    if key not in ("baseline1", "baseline2"):
        return False
    d = dim if dim >= 0 else 3 + dim
    if d == 1:
        blocks = B * triton.cdiv(N, 128)
    elif d == 0:
        blocks = M * triton.cdiv(N, 128)
    else:
        blocks = B * triton.cdiv(M, 128)
    return blocks > 65535


def _run_provider(key, x, dim):
    if key == "torch":
        return _run_torch_ref(x, dim)
    return _model(key, dim)(x)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _bench_once(fn, warmup=5, rep=20):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
            _sync()
        start = time.perf_counter()
        for _ in range(rep):
            fn()
            _sync()
        return (time.perf_counter() - start) * 1000.0 / rep


def unit_test():
    torch.manual_seed(0)
    ok = True
    for label, B, M, N, dim in _TEST_SHAPES + _BENCH_SHAPES:
        x = _make_inputs(B, M, N)
        ref = _run_torch_ref(x, dim)
        for key in ("baseline1", "baseline2", "optimized"):
            _PROVIDER_NAMES[key]
            if _baseline_grid_overflows(key, B, M, N, dim):
                print(f"TEST {key} {label} SKIP grid_guard")
                continue
            try:
                y = _run_provider(key, x, dim)
                _sync()
                torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)
                print(f"TEST {key} {label} PASS")
            except BaseException as e:
                if key == "optimized":
                    ok = False
                    print(
                        f"TEST {key} {label} MISMATCH {type(e).__name__}: {str(e)[:160]}"
                    )
                else:
                    print(
                        f"TEST {key} {label} INFO baseline_unavailable {type(e).__name__}: {str(e)[:160]}"
                    )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


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
        styles=[("black", "-"), ("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="ms",
        plot_name="max_reduction_dim_benchmark",
        args={},
    ))
def benchmark(label, provider):
    shape = {s[0]: s for s in _BENCH_SHAPES}[label]
    _, B, M, N, dim = shape
    if _baseline_grid_overflows(provider, B, M, N, dim):
        return float("inf")
    try:
        x = _make_inputs(B, M, N)

        def fn():
            return _run_provider(provider, x, dim)

        fn()
        _sync()
        return _bench_once(fn, warmup=3, rep=10)
    except BaseException as e:
        print(
            f"INFO benchmark {provider} {label} unavailable {type(e).__name__}: {str(e)[:160]}"
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
        benchmark.run(print_data=True, show_plots=False, save_path=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
