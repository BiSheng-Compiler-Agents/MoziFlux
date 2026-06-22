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
    "baseline1": ("Baseline Triton1", "22_Tanh.py"),
    "baseline2": ("Baseline Triton2", "base_22_Tanh.py"),
    "optimized": ("Optimized Triton", "opt_22_Tanh.py"),
}

_BENCH_SHAPES = [
    ("N4096", 4096),
    ("N1M", 1_000_000),
    ("N10M", 10_000_000),
    ("N16777215", 16_777_215),
    ("N100M", 100_000_000),
    ("benchmark", 4096 * 393216),
    ("N127", 127),
    ("N4097", 4097),
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
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        torch.manual_seed(0)
        # get_init_inputs may return [()] meaning "no args"; flatten appropriately
        if init == [()]:
            init = []
        _MODEL_CACHE[key] = mod.ModelNew(*init).to(device="npu").eval()
    return _MODEL_CACHE[key]


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _make_inputs(n, seed=42, dtype=torch.float16):
    torch.manual_seed(seed)
    return (torch.randn((n, ), device="npu", dtype=dtype) * 5.0).contiguous()


def _run_torch_ref(x):
    return torch.tanh(x)


def _run_provider(key, x):
    if key == "torch":
        return _run_torch_ref(x)
    return _model(key)(x)


def _would_exceed_core_dim(key, n):
    if key in ("torch", "baseline2", "optimized"):
        return False
    # baseline1 direct grid uses fixed BLOCK_SIZE=4096.
    return triton.cdiv(n, 4096) > 65535


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
    for label, n in _BENCH_SHAPES:
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            x = _make_inputs(n, dtype=dtype)
            ref = _run_torch_ref(x)
            _sync()
            for key in ("baseline1", "baseline2", "optimized"):
                try:
                    if _would_exceed_core_dim(key, n):
                        print(
                            f"TEST {label} {key} {dtype}: INFO coreDim would exceed Ascend 65535; benchmark returns inf"
                        )
                        continue
                    out = _run_provider(key, x.clone())
                    _sync()
                    max_abs = (out.float() - ref.float()).abs().max().item()
                    close = torch.allclose(out.float(),
                                           ref.float(),
                                           atol=1e-3,
                                           rtol=1e-3)
                    status = "PASS" if close else ("INFO" if key in (
                        "baseline1", "baseline2") else "MISMATCH")
                    print(
                        f"TEST {label} {key} {dtype}: {status} max_abs={max_abs:.6g}"
                    )
                    if key == "optimized":
                        optimized_ok = optimized_ok and bool(close)
                except Exception as exc:
                    print(
                        f"TEST {label} {key} {dtype}: INFO {type(exc).__name__}: {exc}"
                    )
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
        plot_name="tanh_performance",
        args={},
    ))
def benchmark(label, provider):
    n = {s[0]: s[1] for s in _BENCH_SHAPES}[label]
    x = _make_inputs(n, dtype=torch.float16)
    try:
        if _would_exceed_core_dim(provider, n):
            print(
                f"BENCH {label} {provider}: INFO coreDim would exceed Ascend 65535; returning inf"
            )
            return float("inf")
        return _bench_one(lambda: _run_provider(provider, x))
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
