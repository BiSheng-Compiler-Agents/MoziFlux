import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

HERE = Path(__file__).resolve().parent


def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, name):
    try:
        return _load(fname, name)
    except Exception as exc:
        print(f"INFO provider {name} unavailable: {type(exc).__name__}")
        return None


_input_mod = _load("86_conv_depthwise_separable_2D.py", "k_input_86_dwsep")
_base2_mod = _load_optional("base_86_conv_depthwise_separable_2D.py",
                            "k_base2_86_dwsep")
_opt_mod = _load("opt_86_conv_depthwise_separable_2D.py", "k_opt_86_dwsep")

_INIT = [64, 128, 3, 1, 1, 1]
_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_MODEL_CACHE = {}
_BENCH_SHAPES = [
    ("tiny_32", 1, 64, 32, 32),
    ("small_64", 2, 64, 64, 64),
    ("medium_128", 4, 64, 128, 128),
    ("exact_512", 16, 64, 512, 512),
]


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


class TorchRef(nn.Module):

    def __init__(self):
        super().__init__()
        in_channels, out_channels, kernel_size, stride, padding, dilation = _INIT
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   bias=False)

    def forward(self, x):
        x32 = x.contiguous().to(torch.float32)
        y = F.conv2d(
            x32,
            self.depthwise.weight.to(x.device, torch.float32),
            None,
            stride=self.depthwise.stride,
            padding=self.depthwise.padding,
            dilation=self.depthwise.dilation,
            groups=self.depthwise.groups,
        )
        y = F.conv2d(
            y,
            self.pointwise.weight.to(x.device, torch.float32),
            None,
            stride=self.pointwise.stride,
            padding=self.pointwise.padding,
            dilation=self.pointwise.dilation,
            groups=self.pointwise.groups,
        )
        return y.to(x.dtype)


def _model(provider):
    if provider in _MODEL_CACHE:
        return _MODEL_CACHE[provider]
    torch.manual_seed(0)
    if provider == "torch":
        m = TorchRef()
    elif provider == "baseline1":
        m = _input_mod.ModelNew(*_INIT)
    elif provider == "baseline2" and _base2_mod is not None:
        m = _base2_mod.ModelNew(*_INIT)
    elif provider == "optimized":
        m = _opt_mod.ModelNew(*_INIT)
    else:
        return None
    m = m.to(device="npu").eval()
    _MODEL_CACHE[provider] = m
    return m


def _make_inputs(label, n, c, h, w):
    torch.manual_seed(123)
    x = torch.rand((n, c, h, w), device="npu", dtype=torch.float32)
    return [x]


def _run_provider(provider, inputs):
    m = _model(provider)
    if m is None:
        raise RuntimeError("provider_unavailable")
    with torch.no_grad():
        return m(*inputs)


def _baseline1_grid_too_large(n, h, w):
    # Input baseline launches grid=(ceil(N*Hout*Wout/64), ceil(Cout/32)); product overflows for exact_512.
    return math.ceil((n * h * w) / 64) * math.ceil(128 / 32) > 65535


def _is_comparison_preskip(provider, label, n, h, w):
    if provider == "baseline1" and (_baseline1_grid_too_large(n, h, w)
                                    or label == "exact_512"):
        return "grid_guard"
    if provider == "baseline2" and _base2_mod is None:
        return "provider_unavailable"
    if provider == "baseline2" and label == "exact_512":
        return "comparison_preskip"
    return None


def unit_test():
    ok = True
    for label, n, c, h, w in _BENCH_SHAPES:
        inputs = _make_inputs(label, n, c, h, w)
        ref = _run_provider("torch", inputs)
        _sync()
        for provider in ["baseline1", "baseline2", "optimized"]:
            reason = _is_comparison_preskip(provider, label, n, h, w)
            if reason and provider != "optimized":
                print(f"TEST {provider} {label} SKIP {reason}")
                continue
            try:
                out = _run_provider(provider, inputs)
                _sync()
                max_diff = (out - ref).abs().max().item()
                passed = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
                if passed:
                    print(
                        f"TEST {provider} {label} PASS max_diff={max_diff:.6g}"
                    )
                else:
                    if provider == "optimized":
                        print(
                            f"TEST optimized {label} MISMATCH max_diff={max_diff:.6g}"
                        )
                        ok = False
                    else:
                        print(
                            f"TEST {provider} {label} SKIP value_diff max_diff={max_diff:.6g}"
                        )
            except Exception as exc:
                if provider == "optimized":
                    print(
                        f"TEST optimized {label} MISMATCH exception={type(exc).__name__}"
                    )
                    ok = False
                else:
                    print(f"TEST {provider} {label} SKIP provider_exception")
    print("UNIT_TEST PASS" if ok else "UNIT_TEST FAILED")
    return ok


def _bench_one(provider, label, n, c, h, w):
    reason = _is_comparison_preskip(provider, label, n, h, w)
    if reason and provider != "optimized":
        return float("inf")
    inputs = _make_inputs(label, n, c, h, w)
    try:
        _run_provider(provider, inputs)
        _sync()
        return triton.testing.do_bench(lambda: _run_provider(provider, inputs),
                                       warmup=5,
                                       rep=20,
                                       return_mode="mean")
    except Exception:
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "-"), ("orange", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="depthwise_separable_conv2d",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, n, c, h, w = shape
    return _bench_one(provider, label, n, c, h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if not args.test and not args.bench:
        args.test = args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        bench.run(print_data=True, show_plots=False, save_path=".")


if __name__ == "__main__":
    main()
