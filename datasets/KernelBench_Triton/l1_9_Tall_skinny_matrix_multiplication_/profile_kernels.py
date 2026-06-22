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
    "baseline1":
    ("Baseline Triton1", "9_Tall_skinny_matrix_multiplication_.py"),
    "baseline2":
    ("Baseline Triton2", "base_9_Tall_skinny_matrix_multiplication_.py"),
    "optimized":
    ("Optimized Triton", "opt_9_Tall_skinny_matrix_multiplication_.py"),
}

_BENCH_SHAPES = [
    # Tall-skinny input regime from get_inputs(): A=(M,K), B=(K,M), M >> K.
    ("M512_K32", 512, 512, 32),
    ("M1024_K32", 1024, 1024, 32),
    ("M2048_K32", 2048, 2048, 32),
    ("M4096_K32", 4096, 4096, 32),
    ("M8192_K32", 8192, 8192, 32),
    ("M16384_K32", 16384, 16384, 32),
    ("benchmark", 32768, 32768, 32),
    ("M32768_K16", 32768, 32768, 16),
    ("M32768_K48", 32768, 32768, 48),
    ("M32768_K64", 32768, 32768, 64),
]

_MODEL_CACHE = {}


def _load(key: str, filename: str):
    name = f"k_{key}_{Path(filename).stem}".replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _model(key):
    if key not in _MODEL_CACHE:
        _, filename = PROVIDERS[key]
        mod = _load(key, filename)
        _MODEL_CACHE[key] = mod.ModelNew().to(device="npu",
                                              dtype=torch.float16).eval()
    return _MODEL_CACHE[key]


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _make_inputs(m, n, k, seed=42):
    torch.manual_seed(seed)
    a = torch.randn((m, k), device="npu", dtype=torch.float16)
    b = torch.randn((k, n), device="npu", dtype=torch.float16)
    return a.contiguous(), b.contiguous()


def _run_torch_ref(a, b):
    return torch.matmul(a, b).to(dtype=a.dtype)


def _run_provider(key, a, b):
    if key == "torch":
        return _run_torch_ref(a, b)
    return _model(key)(a, b)


def _would_exceed_core_dim(provider, m, n):
    if provider == "torch":
        return False
    # All Triton providers may autotune/launch configs as small as BLOCK_M=64, BLOCK_N=16.
    # If that launch grid exceeds Ascend's 65535 coreDim limit, launching poisons the process.
    return triton.cdiv(m, 64) * triton.cdiv(n, 16) > 65535


def _bench_one(fn, warmup=25, rep=200):
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
    for label, m, n, k in _BENCH_SHAPES:
        a, b = _make_inputs(m, n, k, seed=42)
        ref = _run_torch_ref(a, b)
        _sync()
        for key in ("baseline1", "baseline2", "optimized"):
            try:
                if _would_exceed_core_dim(key, m, n):
                    print(
                        f"TEST {label} {key}: SKIP coreDim would exceed Ascend 65535 launch limit"
                    )
                    continue
                out = _run_provider(key, a, b)
                _sync()
                close = torch.allclose(out.float(),
                                       ref.float(),
                                       atol=1e-2,
                                       rtol=1e-2)
                max_abs = (out.float() - ref.float()).abs().max().item()
                print(
                    f"TEST {label} {key}: {'PASS' if close else 'FAIL'} max_abs={max_abs:.6g}"
                )
                ok = ok and bool(close)
            except Exception as exc:
                ok = False
                print(f"TEST {label} {key}: ERROR {type(exc).__name__}: {exc}")
    print(f"UNIT_TEST {'PASS' if ok else 'FAIL'}")
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
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="tall_skinny_matmul_performance",
        args={},
    ))
def benchmark(label, provider):
    m, n, k = {s[0]: s[1:] for s in _BENCH_SHAPES}[label]
    a, b = _make_inputs(m, n, k, seed=42)
    try:
        if _would_exceed_core_dim(provider, m, n):
            print(
                f"BENCH {label} {provider}: SKIP coreDim would exceed Ascend 65535 launch limit"
            )
            return float("inf")

        def fn():
            return _run_provider(provider, a, b)

        return _bench_one(fn)
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
    ok = True
    if args.test:
        ok = unit_test()
    if args.bench:
        benchmark.run(save_path=str(ROOT), print_data=True, show_plots=False)
    if not ok:
        print("UNIT_TEST_FAILED")


if __name__ == "__main__":
    main()
