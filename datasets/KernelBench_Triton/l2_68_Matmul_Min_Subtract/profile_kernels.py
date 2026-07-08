import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "68_Matmul_Min_Subtract.py"
OPT_FILE = HERE / "opt_68_Matmul_Min_Subtract.py"
BASE2_FILE = HERE / "base_68_Matmul_Min_Subtract.py"

_BENCH_SHAPES = [
    ("small_irregular", 17, 257, 263),
    ("medium", 64, 2048, 2048),
    ("target", 128, 16384, 16384),
]

_MODEL_CACHE = {}
_LOADED = {}


def _load(path: Path, name: str):
    if name in _LOADED:
        return _LOADED[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _LOADED[name] = mod
    return mod


def _init_args(in_features, out_features):
    return [in_features, out_features, 2.0]


class TorchRef(nn.Module):

    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        c = self.constant.detach().to(device=x.device, dtype=x.dtype)
        return torch.minimum(
            self.linear(x) - c, torch.zeros((), device=x.device,
                                            dtype=x.dtype))


def _model(provider, in_features, out_features):
    key = (provider, in_features, out_features)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    if provider == "torch_ref":
        m = TorchRef(*_init_args(in_features, out_features)).npu().eval()
    elif provider == "baseline1":
        mod = _load(INPUT_FILE, "k_l2_68_input")
        m = mod.ModelNew(*_init_args(in_features, out_features)).npu().eval()
    elif provider == "optimized":
        mod = _load(OPT_FILE, "k_l2_68_opt")
        m = mod.ModelNew(*_init_args(in_features, out_features)).npu().eval()
    else:
        raise ValueError(provider)
    _MODEL_CACHE[key] = m
    return m


def _make_inputs(M, K, dtype=torch.float32):
    torch.manual_seed(123)
    return torch.rand((M, K), device="npu", dtype=dtype)


def _run_provider(provider, x, K, N):
    if provider == "baseline2":
        raise RuntimeError(
            "Baseline Triton2 is a read-only sandbox reference and is not imported by this profiler"
        )
    return _model(provider, K, N)(x)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok = True
    for label, M, K, N in _BENCH_SHAPES:
        x = _make_inputs(M, K)
        with torch.no_grad():
            ref = _run_provider("torch_ref", x, K, N)
            torch.npu.synchronize()
            for provider, display in [
                ("baseline1", "Baseline Triton1"),
                ("baseline2", "Baseline Triton2"),
                ("optimized", "Optimized Triton"),
            ]:
                if provider == "baseline2":
                    print(
                        f"TEST {display} {label}: SKIP_UNAVAILABLE sandbox_do_not_read_base max_abs=inf"
                    )
                    continue
                try:
                    out = _run_provider(provider, x, K, N)
                    torch.npu.synchronize()
                    diff = _max_abs(out, ref)
                    passed = math.isfinite(diff) and diff <= 1e-3
                    print(
                        f"TEST {display} {label}: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}"
                    )
                    ok = ok and (passed if provider == "optimized" else ok)
                except Exception as exc:
                    tag = type(exc).__name__
                    if provider == "optimized":
                        ok = False
                        print(
                            f"TEST {display} {label}: FAIL {tag} max_abs=inf")
                    else:
                        print(
                            f"TEST {display} {label}: SKIP_UNAVAILABLE {tag} max_abs=inf"
                        )
            # cover optimized parameter-cache hit path with a second call
            try:
                out2 = _run_provider("optimized", x, K, N)
                torch.npu.synchronize()
                diff2 = _max_abs(out2, ref)
                hit_ok = math.isfinite(diff2) and diff2 <= 1e-3
                print(
                    f"TEST Optimized Triton {label}_cache_hit: {'PASS' if hit_ok else 'FAIL'} max_abs={diff2:.6g}"
                )
                ok = ok and hit_ok
            except Exception as exc:
                ok = False
                print(
                    f"TEST Optimized Triton {label}_cache_hit: FAIL {type(exc).__name__} max_abs=inf"
                )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_ms(fn):
    if hasattr(triton.testing, "do_bench"):
        return triton.testing.do_bench(fn,
                                       warmup=10,
                                       rep=50,
                                       return_mode="mean")
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / 20.0


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="l2_68_matmul_min_subtract",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, M, K, N = shape
    if provider == "baseline2":
        return float("inf")
    try:
        x = _make_inputs(M, K)
        _run_provider(provider, x, K, N)
        torch.npu.synchronize()
        return _bench_ms(lambda: _run_provider(provider, x, K, N))
    except Exception as exc:
        print(
            f"INFO benchmark_unavailable {provider} {label}: {type(exc).__name__}"
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
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
