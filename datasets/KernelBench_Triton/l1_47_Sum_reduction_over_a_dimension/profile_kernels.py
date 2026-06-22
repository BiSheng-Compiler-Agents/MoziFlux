import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
PROVIDERS = {
    "baseline1": ROOT / "47_Sum_reduction_over_a_dimension.py",
    "baseline2": ROOT / "base_47_Sum_reduction_over_a_dimension.py",
    "optimized": ROOT / "opt_47_Sum_reduction_over_a_dimension.py",
}
_BENCH_SHAPES = [
    ("dim1_target", 128, 4096, 4095, 1),
    ("dim1_irregular", 3, 257, 255, 1),
    ("dim0_small", 7, 129, 65, 0),
    ("dim2_small", 5, 130, 257, 2),
]
_MODULES = {}
_MODELS = {}


def _load(key):
    if key in _MODULES:
        return _MODULES[key]
    path = PROVIDERS[key]
    name = f"k_{key}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULES[key] = mod
    return mod


def _model(key, dim):
    cache_key = (key, dim)
    if cache_key in _MODELS:
        return _MODELS[cache_key]
    mod = _load(key)
    try:
        model = mod.ModelNew(dim)
    except TypeError:
        init = mod.get_init_inputs() if hasattr(mod,
                                                "get_init_inputs") else [dim]
        if init == [()]:
            init = []
        if init == [1] and dim != 1:
            init = [dim]
        model = mod.ModelNew(*init)
    model = model.to(device="npu").eval()
    _MODELS[cache_key] = model
    return model


def _make_input(B, M, N):
    torch.manual_seed(0)
    return torch.rand((B, M, N), device="npu")


def _run_torch_ref(x, dim):
    return torch.sum(x, dim=dim, keepdim=True)


def _run_provider(key, x, dim):
    return _model(key, dim)(x)


def _sync():
    torch.npu.synchronize()


def _check_one(key, label, B, M, N, dim):
    x = _make_input(B, M, N)
    ref = _run_torch_ref(x, dim)
    _sync()
    try:
        out = _run_provider(key, x, dim)
        _sync()
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        print(f"TEST {key} {label} PASS")
        return True
    except Exception as exc:
        print(f"TEST {key} {label} INFO_EXCEPTION {type(exc).__name__}")
        return key != "optimized"


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label, B, M, N, dim = shape
        for key in ("baseline1", "baseline2", "optimized"):
            ok = _check_one(key, label, B, M, N, dim) and ok
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_callable(provider, x, dim):
    if provider == "torch":
        return lambda: _run_torch_ref(x, dim)
    return lambda: _run_provider(provider, x, dim)


def _bench_ms(provider, label):
    _, B, M, N, dim = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = _make_input(B, M, N)
    try:
        fn = _bench_callable(provider, x, dim)
        fn()
        _sync()
        return triton.testing.do_bench(fn, warmup=2, rep=5, return_mode="mean")
    except Exception as exc:
        print(f"INFO bench {provider} {label} {type(exc).__name__}")
        return float("inf")


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
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="sum_reduction_dim_benchmark",
        args={},
    ))
def benchmark(label, provider):
    return _bench_ms(provider, label)


def run_bench():
    benchmark.run(print_data=True, show_plots=False)


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
