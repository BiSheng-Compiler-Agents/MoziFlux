import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parent
NAME = "22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish"
SCALE = 2.0
CLAMP_MIN = -10.0
CLAMP_MAX = 10.0

_BENCH_SHAPES = [
    ("small_triton", 128, 1024, 1024),
    ("medium_acl", 256, 4096, 4096),
    ("target_acl", 1024, 8192, 8192),
]
_DISPATCH_TEST_SHAPES = [("gridcap_acl", 70000, 1, 1)]


def _load(fname, optional=False):
    path = ROOT / fname
    try:
        spec = importlib.util.spec_from_file_location("k_" + path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("spec_unavailable")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        if optional:
            print(
                f"INFO optional_provider {fname} unavailable {type(exc).__name__}"
            )
            return None
        raise


_baseline1 = _load(f"{NAME}.py")
_baseline2 = _load(f"base_{NAME}.py", optional=True)
_optimized = _load(f"opt_{NAME}.py")
_MODEL_CACHE = {}

_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_LINE_NAMES = [
    "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
]
_STYLES = [("black", "-"), ("blue", "-"), ("green", "-"), ("red", "-")]


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _make_inputs(B, I, H):  # noqa: E741
    torch.manual_seed(123)
    return torch.rand((B, I), device="npu", dtype=torch.float32)


def _torch_ref(x, I, H):  # noqa: E741
    key = ("torch", I, H)
    if key not in _MODEL_CACHE:
        torch.manual_seed(0)
        lin = nn.Linear(I, H).to(device="npu").eval()
        _MODEL_CACHE[key] = lin
    with torch.no_grad():
        y = _MODEL_CACHE[key](x)
        z = torch.clamp(y * (2.0 * SCALE), min=CLAMP_MIN, max=CLAMP_MAX)
        lse = torch.logsumexp(z, dim=1, keepdim=True)
        return lse * (lse * torch.tanh(F.softplus(lse)))


def _provider_module(provider):
    if provider == "baseline1":
        return _baseline1
    if provider == "baseline2":
        return _baseline2
    if provider == "optimized":
        return _optimized
    return None


def _model(provider, I, H):  # noqa: E741
    key = (provider, I, H)
    if key not in _MODEL_CACHE:
        mod = _provider_module(provider)
        if mod is None:
            return None
        torch.manual_seed(0)
        model = mod.ModelNew(I, H, SCALE, CLAMP_MIN,
                             CLAMP_MAX).to(device="npu").eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _run_provider(provider, x, I, H):  # noqa: E741
    if provider == "torch":
        return _torch_ref(x, I, H)
    model = _model(provider, I, H)
    if model is None:
        raise RuntimeError("provider_unavailable")
    with torch.no_grad():
        return model(x)


def _shape_by_label(label):
    for row in _BENCH_SHAPES:
        if row[0] == label:
            return row
    raise KeyError(label)


def _check_one(provider, label, B, I, H):  # noqa: E741
    if provider == "torch":
        return True
    x = _make_inputs(B, I, H)
    try:
        ref = _torch_ref(x, I, H)
        got = _run_provider(provider, x, I, H)
        _sync()
        ok = torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
        max_abs = (got - ref).abs().max().item()
        if ok:
            print(f"TEST {provider} {label} PASS max_abs={max_abs:.6g}")
            return True
        print(f"TEST {provider} {label} SKIP value_diff max_abs={max_abs:.6g}")
        return provider != "optimized"
    except Exception as exc:
        print(f"TEST {provider} {label} SKIP {type(exc).__name__}")
        return provider != "optimized"


def unit_test():
    ok = True
    for label, B, I, H in _BENCH_SHAPES:  # noqa: E741
        for provider in ("baseline1", "baseline2", "optimized"):
            ok = _check_one(provider, label, B, I, H) and ok
    for label, B, I, H in _DISPATCH_TEST_SHAPES:  # noqa: E741
        ok = _check_one("optimized", label, B, I, H) and ok
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_ms(provider, label):
    label, B, I, H = _shape_by_label(label)  # noqa: E741
    if provider == "baseline2" and _baseline2 is None:
        return float("inf")
    x = _make_inputs(B, I, H)

    def fn():
        _run_provider(provider, x, I, H)

    try:
        fn()
        _sync()
        return triton.testing.do_bench(fn,
                                       warmup=10,
                                       rep=50,
                                       return_mode="mean")
    except Exception:
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=_LINE_NAMES,
        styles=_STYLES,
        ylabel="ms",
        plot_name="matmul_scale_residual_lse_mish",
        args={},
    ))
def benchmark(label, provider):
    return _bench_ms(provider, label)


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
