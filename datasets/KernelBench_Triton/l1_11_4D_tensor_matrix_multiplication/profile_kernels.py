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
    "baseline1": ("Baseline Triton1", "11_4D_tensor_matrix_multiplication.py"),
    "baseline2":
    ("Baseline Triton2", "base_11_4D_tensor_matrix_multiplication.py"),
    "optimized":
    ("Optimized Triton", "opt_11_4D_tensor_matrix_multiplication.py"),
}

_BENCH_SHAPES = [
    # label, B, I, J, W, Kdim. Includes original get_inputs(): B=8,I=256,J=512,W=256,K=768.
    ("B1_small", 1, 64, 128, 64, 192),
    ("B4_medium", 4, 256, 512, 256, 768),
    ("benchmark", 8, 256, 512, 256, 768),
    ("B16_mid", 16, 128, 256, 128, 384),
    ("B32_small", 32, 64, 128, 64, 192),
    ("B1_nonpow2", 1, 123, 200, 97, 300),
    ("B4_large", 4, 512, 1024, 512, 1024),
]

_MODEL_CACHE = {}
_INVALID_CELLS = set()


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
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        _MODEL_CACHE[key] = mod.ModelNew(*init).to(device="npu").eval()
    return _MODEL_CACHE[key]


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _make_inputs(bsz, i, j, w, kdim, seed=42):
    torch.manual_seed(seed)
    a = torch.rand((bsz, i, j, w), device="npu", dtype=torch.float16)
    b = torch.rand((w, kdim), device="npu", dtype=torch.float16)
    return a.contiguous(), b.contiguous()


def _run_torch_ref(a, b):
    bsz, i, j, w = a.shape
    _, kdim = b.shape
    return torch.matmul(a.reshape(-1, w).float(),
                        b.float()).half().view(bsz, i, j, kdim)


def _run_provider(key, a, b):
    if key == "torch":
        return _run_torch_ref(a, b)
    return _model(key)(a, b)


def _would_exceed_core_dim(key, bsz, i, j, kdim, w):
    if key == "torch":
        return False
    m_flat = bsz * i * j
    # baseline1 only has autotuned 2D grid configs; smallest BLOCK_M/N is 64x64.
    if key == "baseline1":
        return triton.cdiv(m_flat, 64) * triton.cdiv(kdim, 64) > 65535
    # baseline2/optimized have target fast paths for the required K=256,N=768 regime;
    # otherwise their generic autotune path also has 64x64 minimum tiles.
    target_path = (w == 256 and kdim == 768 and m_flat % 128 == 0)
    if target_path:
        return False
    return triton.cdiv(m_flat, 64) * triton.cdiv(kdim, 64) > 65535


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
    for label, bsz, i, j, w, kdim in _BENCH_SHAPES:
        a, b = _make_inputs(bsz, i, j, w, kdim)
        ref = _run_torch_ref(a, b)
        _sync()
        for key in ("baseline1", "baseline2", "optimized"):
            try:
                if _would_exceed_core_dim(key, bsz, i, j, kdim, w):
                    _INVALID_CELLS.add((key, label))
                    print(
                        f"TEST {label} {key}: INFO coreDim would exceed Ascend 65535; benchmark returns inf"
                    )
                    continue
                out = _run_provider(key, a, b)
                _sync()
                max_abs = (out.float() - ref.float()).abs().max().item()
                close = torch.allclose(out.float(),
                                       ref.float(),
                                       atol=1.0,
                                       rtol=1e-2)
                status = "PASS" if close else ("INFO" if key in (
                    "baseline1", "baseline2") else "MISMATCH")
                print(f"TEST {label} {key}: {status} max_abs={max_abs:.6g}")
                if not close and key in ("baseline1", "baseline2"):
                    _INVALID_CELLS.add((key, label))
                if key == "optimized":
                    optimized_ok = optimized_ok and bool(close)
            except Exception as exc:
                _INVALID_CELLS.add((key, label))
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
        plot_name="4D_tensor_matmul_perf",
        args={},
    ))
def benchmark(label, provider):
    _, bsz, i, j, w, kdim = next(s for s in _BENCH_SHAPES if s[0] == label)
    a, b = _make_inputs(bsz, i, j, w, kdim)
    try:
        if (provider, label) in _INVALID_CELLS:
            print(
                f"BENCH {label} {provider}: INFO correctness failed or launch invalid; returning inf"
            )
            return float("inf")
        if _would_exceed_core_dim(provider, bsz, i, j, kdim, w):
            print(
                f"BENCH {label} {provider}: INFO coreDim would exceed Ascend 65535; returning inf"
            )
            return float("inf")
        return _bench_one(lambda: _run_provider(provider, a, b))
    except Exception as exc:
        print(f"BENCH {label} {provider}: INFO {type(exc).__name__}: {exc}")
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
