import argparse
import importlib.util
import time
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent

PROVIDERS = {
    "torch": ("PyTorch / ACL", None),
    "baseline1": ("Baseline Triton1", "7_Matmul_with_small_K_dimension_.py"),
    "baseline2":
    ("Baseline Triton2", "base_7_Matmul_with_small_K_dimension_.py"),
    "opt": ("Optimized Triton", "opt_7_Matmul_with_small_K_dimension_.py"),
}
_MODEL_CACHE = {}


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(f"k_{name}", ROOT / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BENCH_SHAPES = [
    ("small", 512, 64, 512),
    ("nonpow2", 777, 37, 913),
    ("medium", 2048, 64, 2048),
    ("benchmark", 32768, 64, 32768),
]


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _device():
    return torch.device("npu")


def _make_inputs(m, k, n):
    torch.manual_seed(0)
    dev = _device()
    a = torch.randn((m, k), device=dev, dtype=torch.float32)
    b = torch.randn((k, n), device=dev, dtype=torch.float32)
    return a, b


def _model(key):
    if key not in _MODEL_CACHE:
        _, filename = PROVIDERS[key]
        mod = _load(key, filename)
        _MODEL_CACHE[key] = mod.ModelNew().to(_device())
    return _MODEL_CACHE[key]


def _run_torch_ref(a, b):
    return torch.matmul(a, b)


def _run_provider(key, a, b):
    if key == "torch":
        return _run_torch_ref(a, b)
    return _model(key)(a, b)


def _skip_invalid_launch(label, provider):
    # baseline1 autotune includes configs whose 32768x32768 grid exceeds Ascend coreDim=65535.
    # Launching it poisons the NPU context, so skip before any kernel/autotune attempt.
    return label == "benchmark" and provider == "baseline1"


def _bench_ms(fn, warmup=10, rep=50):
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


def unit_test():
    ok = True
    for label, m, k, n in _BENCH_SHAPES:
        a, b = _make_inputs(m, k, n)
        ref = _run_torch_ref(a, b)
        _sync()
        for key in ("baseline1", "baseline2", "opt"):
            try:
                if _skip_invalid_launch(label, key):
                    print(
                        "TEST benchmark baseline1: SKIP coreDim would exceed Ascend 65535 launch limit"
                    )
                    continue
                got = _run_provider(key, a, b)
                _sync()
                close = torch.allclose(got, ref, rtol=1e-3, atol=2e-2)
                max_abs = (got - ref).abs().max().item()
                print(
                    f"TEST {label} {key}: {'PASS' if close else 'FAIL'} max_abs={max_abs:.6g}"
                )
                ok = ok and bool(close)
            except Exception as exc:
                ok = False
                print(f"TEST {label} {key}: ERROR {type(exc).__name__}: {exc}")
    return ok


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "baseline1", "baseline2", "opt"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="matmul_small_k_performance",
        args={},
    ))
def benchmark(label, provider):
    shape = {s[0]: s[1:] for s in _BENCH_SHAPES}[label]
    m, k, n = shape
    a, b = _make_inputs(m, k, n)
    try:
        if _skip_invalid_launch(label, provider):
            print(
                "BENCH benchmark baseline1: SKIP coreDim would exceed Ascend 65535 launch limit"
            )
            return float("inf")

        def fn():
            return _run_provider(provider, a, b)

        return _bench_ms(fn)
    except Exception as exc:
        print(f"BENCH {label} {provider}: ERROR {type(exc).__name__}: {exc}")
        return float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not args.test and not args.bench:
        args.test = args.bench = True
    if args.test:
        ok = unit_test()
        print(f"UNIT_TEST {'PASS' if ok else 'FAIL'}")
    if args.bench:
        benchmark.run(print_data=True,
                      show_plots=False,
                      save_path=str(ROOT / "results"))


if __name__ == "__main__":
    main()
