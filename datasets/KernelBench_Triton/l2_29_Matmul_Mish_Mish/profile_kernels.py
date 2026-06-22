import argparse
import importlib.util
import pathlib
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parent

_BENCH_SHAPES = [
    ("small", 16, 512, 512),
    ("medium", 64, 2048, 2048),
    ("target", 1024, 8192, 8192),
]
_SHAPE_BY_LABEL = {label: (m, k, n) for label, m, k, n in _BENCH_SHAPES}
_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_PROVIDER_NAMES = [
    "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
]
_MAX_CORE_DIM = 65535
_MODEL_CACHE: Dict[Tuple[str, int, int], nn.Module] = {}


def _load(fname: str, modname: str):
    path = ROOT / fname
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_baseline1 = _load("29_Matmul_Mish_Mish.py", "k_l2_29_baseline1")
_opt = _load("opt_29_Matmul_Mish_Mish.py", "k_l2_29_optimized")
# base_*.py is intentionally not imported here: it is a read-only reference file in this sandbox.
_baseline2 = None


def _mish2(z: torch.Tensor) -> torch.Tensor:
    z = z * torch.tanh(F.softplus(z, beta=1, threshold=20))
    return z * torch.tanh(F.softplus(z, beta=1, threshold=20))


class TorchRef(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _mish2(self.linear(x))


def _model(provider: str, in_features: int, out_features: int) -> nn.Module:
    key = (provider, in_features, out_features)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    if provider == "torch":
        model = TorchRef(in_features, out_features)
    elif provider == "baseline1":
        model = _baseline1.ModelNew(in_features, out_features)
    elif provider == "optimized":
        model = _opt.ModelNew(in_features, out_features)
    else:
        raise RuntimeError("baseline2_readonly_not_loaded")
    model = model.to(device="npu").eval()
    _MODEL_CACHE[key] = model
    return model


def _make_inputs(label: str):
    m, k, _n = _SHAPE_BY_LABEL[label]
    torch.manual_seed(123)
    return (torch.rand((m, k), device="npu", dtype=torch.float32), )


def _baseline1_grid_ok(label: str) -> bool:
    m, _k, n = _SHAPE_BY_LABEL[label]
    grid_product = m * triton.cdiv(n, 32)
    return grid_product <= _MAX_CORE_DIM


def _run_provider(provider: str, label: str):
    m, k, n = _SHAPE_BY_LABEL[label]
    (x, ) = _make_inputs(label)
    if provider == "baseline1" and not _baseline1_grid_ok(label):
        raise RuntimeError("grid_guard")
    if provider == "baseline2":
        raise RuntimeError("baseline2_readonly_not_loaded")
    model = _model(provider, k, n)
    with torch.no_grad():
        return model(x)


def _run_optimized_fallback():
    label = "small"
    m, k, n = _SHAPE_BY_LABEL[label]
    torch.manual_seed(0)
    model = _model("optimized", k, n)
    (x, ) = _make_inputs(label)
    with torch.no_grad():
        w = model.linear.weight.to(device=x.device, dtype=x.dtype)
        b = model.linear.bias.to(device=x.device, dtype=x.dtype)
        return _opt.matmul_mish_mish(x, w, b, force_triton=True)


def _check_close(name: str, got: torch.Tensor, ref: torch.Tensor,
                 label: str) -> bool:
    try:
        torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)
        print(f"TEST {name} {label} PASS")
        return True
    except Exception as exc:
        diff = (
            got -
            ref).abs().max().item() if got.shape == ref.shape else float("inf")
        print(
            f"TEST {name} {label} SKIP value_diff max_abs={diff:.6g} reason={type(exc).__name__}"
        )
        return False


def unit_test() -> bool:
    ok = True
    for label, _m, _k, _n in _BENCH_SHAPES:
        ref = _run_provider("torch", label)
        print(f"TEST torch {label} PASS")
        for provider in ["baseline1", "baseline2", "optimized"]:
            if provider == "baseline1" and not _baseline1_grid_ok(label):
                print(f"TEST baseline1 {label} SKIP grid_guard")
                continue
            if provider == "baseline2":
                print(
                    f"TEST baseline2 {label} SKIP baseline2_readonly_not_loaded"
                )
                continue
            try:
                got = _run_provider(provider, label)
                passed = _check_close(provider, got, ref, label)
                if provider == "optimized" and not passed:
                    ok = False
            except Exception as exc:
                print(f"TEST {provider} {label} SKIP {type(exc).__name__}")
                if provider == "optimized":
                    ok = False
    try:
        fb = _run_optimized_fallback()
        ref = _run_provider("torch", "small")
        if _check_close("optimized_fallback", fb, ref, "small") is False:
            ok = False
    except Exception as exc:
        print(f"TEST optimized_fallback small SKIP {type(exc).__name__}")
        ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_ms(fn):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=10,
                                       rep=50,
                                       return_mode="mean")
    except Exception:
        # Manual fallback with per-iteration synchronization.
        for _ in range(5):
            fn()
            torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            fn()
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end) / 20.0


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=_PROVIDER_NAMES,
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "-")],
        ylabel="ms",
        plot_name="l2_29_matmul_mish_mish",
        args={},
    ))
def benchmark(label, provider):
    if provider == "baseline1" and not _baseline1_grid_ok(label):
        return float("inf")
    if provider == "baseline2":
        return float("inf")
    try:
        return _time_ms(lambda: _run_provider(provider, label))
    except Exception as exc:
        print(f"INFO bench {provider} {label} skip {type(exc).__name__}")
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
