import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

try:
    import triton
except Exception:
    triton = None

ROOT = Path(__file__).resolve().parent


def _load(fname, name):
    path = ROOT / fname
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


baseline1_mod = _load("32_Conv2d_Scaling_Min.py",
                      "k_baseline1_32_conv2d_scaling_min")
opt_mod = _load("opt_32_Conv2d_Scaling_Min.py",
                "k_optimized_32_conv2d_scaling_min")

_BENCH_SHAPES = [
    ("small_32", 4, 64, 32, 32),
    ("medium_128", 16, 64, 128, 128),
    ("exact_256", 64, 64, 256, 256),
]
_SHAPES = {s[0]: s[1:] for s in _BENCH_SHAPES}
_DEFAULT_INIT = (64, 128, 3, 2.0)
_MODEL_CACHE = {}


class TorchRef(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        y = F.conv2d(x,
                     self.conv.weight,
                     self.conv.bias,
                     stride=self.conv.stride,
                     padding=self.conv.padding,
                     dilation=self.conv.dilation,
                     groups=self.conv.groups)
        if self.scale_factor >= 0.0:
            return torch.amin(y, dim=1, keepdim=True).mul(self.scale_factor)
        return torch.amax(y, dim=1, keepdim=True).mul(self.scale_factor)


def _device():
    return "npu" if hasattr(torch,
                            "npu") and torch.npu.is_available() else "cpu"


def _sync():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def _model(provider, init_args=_DEFAULT_INIT, force_triton=False):
    key = (provider, tuple(init_args), bool(force_triton))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    if provider == "torch":
        m = TorchRef(*init_args)
    elif provider == "baseline1":
        m = baseline1_mod.ModelNew(*init_args)
    elif provider == "optimized":
        try:
            m = opt_mod.ModelNew(*init_args, force_triton=force_triton)
        except TypeError:
            m = opt_mod.ModelNew(*init_args)
    else:
        return None
    m = m.to(device=_device()).eval()
    _MODEL_CACHE[key] = m
    return m


def _make_input(B, C, H, W):
    torch.manual_seed(123)
    x = torch.rand((B, C, H, W), dtype=torch.float32)
    return x.to(_device())


def _run(provider, x, init_args=_DEFAULT_INIT, force_triton=False):
    if provider == "baseline2":
        return None
    m = _model(provider, init_args, force_triton=force_triton)
    with torch.no_grad():
        return m(x)


def _check_one(provider,
               label,
               init_args=_DEFAULT_INIT,
               shape=None,
               force_triton=False):
    if provider == "baseline2":
        print(f"TEST baseline2 {label} SKIP baseline2_readonly_not_loaded")
        return True
    if shape is None:
        shape = _SHAPES[label]
    x = _make_input(*shape)
    try:
        ref = _run("torch", x, init_args)
        got = _run(provider, x, init_args, force_triton=force_triton)
        _sync()
        max_abs = (got - ref).abs().max().item()
        ok = bool(torch.allclose(got, ref, rtol=1e-3, atol=1e-3))
        status = "PASS" if ok else "MISMATCH"
        print(f"TEST {provider} {label} {status} max_abs={max_abs:.6g}")
        return ok if provider == "optimized" else True
    except Exception as exc:
        print(
            f"TEST {provider} {label} SKIP provider_unavailable {type(exc).__name__}: {str(exc)[:120]}"
        )
        return provider != "optimized"


def unit_test():
    ok = True
    for label, *_ in _BENCH_SHAPES:
        for provider in ("baseline1", "baseline2", "optimized"):
            ok = _check_one(provider, label) and ok
    # Dispatch coverage for optimized custom Triton direct and persistent epilogues.
    ok = _check_one("optimized",
                    "triton_direct_dispatch",
                    init_args=(1, 16, 1, 2.0),
                    shape=(8, 1, 4, 4),
                    force_triton=True) and ok
    ok = _check_one("optimized",
                    "triton_persistent_dispatch",
                    init_args=(1, 2, 1, 2.0),
                    shape=(70000, 1, 1, 1),
                    force_triton=True) and ok
    print("UNIT_TEST PASS" if ok else "UNIT_TEST FAIL")
    return ok


def _time_provider(provider, label):
    if provider == "baseline2":
        print(
            f"INFO benchmark baseline2 {label} inf baseline2_readonly_not_loaded"
        )
        return float("inf")
    B, C, H, W = _SHAPES[label]
    x = _make_input(B, C, H, W)

    # Avoid expensive comparison-provider cells poisoning optimized timing if unavailable.
    def fn():
        with torch.no_grad():
            y = _run(provider, x)
        return y

    try:
        fn()
        _sync()
        if triton is not None and hasattr(triton, "testing") and hasattr(
                triton.testing, "do_bench"):
            return triton.testing.do_bench(fn,
                                           warmup=3,
                                           rep=10,
                                           return_mode="mean")
        times = []
        for _ in range(3):
            fn()
            _sync()
        for _ in range(10):
            t0 = time.perf_counter()
            fn()
            _sync()
            times.append((time.perf_counter() - t0) * 1000.0)
        return sum(times) / len(times)
    except Exception as exc:
        print(
            f"INFO benchmark {provider} {label} inf provider_unavailable {type(exc).__name__}: {str(exc)[:120]}"
        )
        return float("inf")


if triton is not None:

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
            styles=[("blue", "-"), ("red", "-"), ("black", "--"),
                    ("green", "-")],
            ylabel="Latency (ms)",
            plot_name="conv2d_scaling_min",
            args={},
        ))
    def bench(label, provider):
        return _time_provider(provider, label)
else:
    bench = None


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
        if bench is not None:
            bench.run(print_data=True, show_plots=False)
        else:
            print(
                "label  PyTorch / ACL  Baseline Triton1  Baseline Triton2  Optimized Triton"
            )
            for label, *_ in _BENCH_SHAPES:
                vals = [
                    _time_provider(p, label)
                    for p in ("torch", "baseline1", "baseline2", "optimized")
                ]
                print(label, *vals)


if __name__ == "__main__":
    main()
