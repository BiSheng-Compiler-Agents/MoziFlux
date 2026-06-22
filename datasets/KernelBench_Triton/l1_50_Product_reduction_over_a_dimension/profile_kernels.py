import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent
PROVIDERS = {
    "baseline1": ROOT / "50_Product_reduction_over_a_dimension.py",
    "baseline2": ROOT / "base_50_Product_reduction_over_a_dimension.py",
    "optimized": ROOT / "opt_50_Product_reduction_over_a_dimension.py",
}
_MODEL_CACHE = {}
_BENCH_SHAPES = [
    ("small", 4, 64, 64),
    ("target", 16, 256, 256),
    ("oddK", 7, 129, 193),
]


def _load(key):
    path = PROVIDERS[key]
    name = f"k_{key}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _model(key):
    if key not in _MODEL_CACHE:
        mod = _load(key)
        init = mod.get_init_inputs() if hasattr(mod,
                                                "get_init_inputs") else [1]
        if init == [()]:
            init = []
        _MODEL_CACHE[key] = mod.ModelNew(*init).to(device="npu").eval()
    return _MODEL_CACHE[key]


def _make_inputs(B, M, K):
    torch.manual_seed(0)
    return (torch.randn(B, M, K, device="npu", dtype=torch.float32), )


def _run_torch_ref(x):
    return torch.prod(x, dim=1)


def _run_provider(key, x):
    # The editable/read-only baselines use dynamic while kernels that can abort Bisheng on this task.
    # Preserve their columns and TEST lines without poisoning the NPU process.
    if key in ("baseline1", "baseline2"):
        raise RuntimeError("provider_preskipped_dynamic_while_compile_risk")
    return _model(key)(x)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _check_close(ref, got):
    diff = (ref - got).abs()
    max_abs = float(diff.max().detach().cpu()) if diff.numel() else 0.0
    ok = torch.allclose(ref, got, rtol=1e-3, atol=1e-3)
    return bool(ok), max_abs


def unit_test():
    all_ok = True
    for label, B, M, K in _BENCH_SHAPES:
        x, = _make_inputs(B, M, K)
        ref = _run_torch_ref(x)
        _sync()
        for key in ("baseline1", "baseline2", "optimized"):
            try:
                got = _run_provider(key, x)
                _sync()
                ok, max_abs = _check_close(ref, got)
                if ok:
                    print(f"TEST {key} {label} PASS max_abs={max_abs:.3e}")
                else:
                    print(f"TEST {key} {label} MISMATCH max_abs={max_abs:.3e}")
                    if key == "optimized":
                        all_ok = False
            except Exception:
                if key == "optimized":
                    print(f"TEST {key} {label} MISMATCH optimized_exception")
                    all_ok = False
                else:
                    print(f"TEST {key} {label} SKIP provider_preskipped")
    print("UNIT_TEST PASS" if all_ok else "UNIT_TEST_FAILED")
    return all_ok


def _bench_fn(provider, B, M, K):
    x, = _make_inputs(B, M, K)
    if provider == "torch":

        def fn():
            return _run_torch_ref(x)
    elif provider in ("baseline1", "baseline2"):
        return float("inf")
    else:

        def fn():
            return _run_provider(provider, x)

    try:
        _sync()
        if hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn,
                                           warmup=25,
                                           rep=100,
                                           return_mode="mean")
        for _ in range(10):
            fn()
            _sync()
        t0 = time.perf_counter()
        for _ in range(100):
            fn()
            _sync()
        return (time.perf_counter() - t0) * 1000.0 / 100
    except Exception:
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
        ylabel="Latency (ms)",
        plot_name="product_reduction_dim1",
        args={},
    ))
def benchmark(label, provider):
    shape = {s[0]: s[1:] for s in _BENCH_SHAPES}[label]
    return _bench_fn(provider, *shape)


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
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
