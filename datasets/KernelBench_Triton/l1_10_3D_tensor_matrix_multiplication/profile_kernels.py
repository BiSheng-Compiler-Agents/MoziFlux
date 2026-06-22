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
    "baseline1": ("Baseline Triton1", "10_3D_tensor_matrix_multiplication.py"),
    "baseline2":
    ("Baseline Triton2", "base_10_3D_tensor_matrix_multiplication.py"),
    "optimized":
    ("Optimized Triton", "opt_10_3D_tensor_matrix_multiplication.py"),
}

_BENCH_SHAPES = [
    # label, B, M, N, K. Includes original get_inputs(): B=16, M=1024, K=2048, N=768.
    ("B1_small", 1, 128, 128, 256),
    ("B4_small", 4, 128, 128, 256),
    ("B1_medium", 1, 512, 512, 512),
    ("B4_medium", 4, 512, 512, 512),
    ("B16_medium", 16, 256, 256, 256),
    ("B8_tallM", 8, 1024, 512, 256),
    ("B8_wideN", 8, 512, 1024, 256),
    ("B2_nonpow2", 2, 768, 768, 768),
    ("B4_nonpow2", 4, 640, 832, 576),
    ("B8_large", 8, 512, 512, 512),
    ("B1_benchmark", 1, 1024, 768, 2048),
    ("benchmark", 16, 1024, 768, 2048),
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
        torch.manual_seed(0)
        _MODEL_CACHE[key] = mod.ModelNew().to(device="npu").eval()
    return _MODEL_CACHE[key]


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _make_inputs(bsz, m, n, k, seed=42):
    torch.manual_seed(seed)
    a = torch.rand((bsz, m, k), device="npu", dtype=torch.float16)
    b = torch.rand((k, n), device="npu", dtype=torch.float16)
    return a.contiguous(), b.contiguous()


def _run_torch_ref(a, b):
    return torch.matmul(a.float(), b.float()).half()


def _run_provider(key, a, b):
    if key == "torch":
        return _run_torch_ref(a, b)
    return _model(key)(a, b)


def _known_provider_inf(key):
    # base_10 currently MLIR-fails for every tested shape (UB overflow/vcast). Keep the
    # Baseline Triton2 column, but do not launch it and spam verifier output.
    return key == "baseline2"


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
    optimized_ok = True
    for label, bsz, m, n, k in _BENCH_SHAPES:
        a, b = _make_inputs(bsz, m, n, k)
        ref = _run_torch_ref(a, b)
        _sync()
        for key in ("baseline1", "baseline2", "optimized"):
            try:
                if _known_provider_inf(key):
                    print(
                        f"TEST {label} {key}: INFO known MLIR compile issue; benchmark returns inf"
                    )
                    continue
                out = _run_provider(key, a, b)
                _sync()
                max_abs = (out.float() - ref.float()).abs().max().item()
                close = torch.allclose(out.float(),
                                       ref.float(),
                                       atol=1.0,
                                       rtol=1e-2)
                status = "PASS" if close else (
                    "INFO" if key == "baseline1" else "MISMATCH")
                print(f"TEST {label} {key}: {status} max_abs={max_abs:.6g}")
                if key == "optimized":
                    optimized_ok = optimized_ok and bool(close)
            except Exception as exc:
                print(f"TEST {label} {key}: INFO {type(exc).__name__}: {exc}")
                if key == "optimized":
                    optimized_ok = False

    print(f"UNIT_TEST {'PASS' if optimized_ok else 'FAIL'}")
    return optimized_ok


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
        plot_name="3D_tensor_matmul_perf",
        args={},
    ))
def benchmark(label, provider):
    _, bsz, m, n, k = next(s for s in _BENCH_SHAPES if s[0] == label)
    a, b = _make_inputs(bsz, m, n, k)
    try:
        if _known_provider_inf(provider):
            print(
                f"BENCH {label} {provider}: SKIP known MLIR compile failure; returning inf"
            )
            return float("inf")
        return _bench_one(lambda: _run_provider(provider, a, b))
    except Exception as exc:
        print(f"BENCH {label} {provider}: SKIP {type(exc).__name__}: {exc}")
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
