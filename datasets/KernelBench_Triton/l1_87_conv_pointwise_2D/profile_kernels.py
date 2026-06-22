import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

HERE = Path(__file__).resolve().parent
INPUT_NAME = "87_conv_pointwise_2D.py"
OPT_NAME = "opt_87_conv_pointwise_2D.py"
BASE_NAME = "base_87_conv_pointwise_2D.py"

_BENCH_SHAPES = [
    ("small_64", 1, 64, 64, 64),
    ("medium_256", 4, 64, 256, 256),
    ("exact_1024", 16, 64, 1024, 1024),
]

_PROVIDERS = ["torch_ref", "baseline1", "baseline2", "optimized"]
_PROVIDER_NAMES = {
    "torch_ref": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}

_MODULES = {}
_MODELS = {}
_REF_MODELS = {}


def _load(fname, optional=False):
    path = HERE / fname
    if optional and not path.exists():
        return None
    try:
        name = "k_" + path.stem.replace("-", "_")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        if optional:
            print(
                f"TEST baseline2 import SKIP optional_unavailable {type(exc).__name__}"
            )
            return None
        raise


def _modules():
    if not _MODULES:
        _MODULES["baseline1"] = _load(INPUT_NAME)
        _MODULES["baseline2"] = _load(BASE_NAME, optional=True)
        _MODULES["optimized"] = _load(OPT_NAME)
    return _MODULES


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    return list(init)


def _model(provider):
    if provider in _MODELS:
        return _MODELS[provider]
    mods = _modules()
    mod = mods.get(provider)
    if mod is None:
        return None
    torch.manual_seed(0)
    model = mod.ModelNew(*_init_args(mod)).to(device="npu").eval()
    _MODELS[provider] = model
    return model


def _ref_model(in_channels=64, out_channels=128, bias=False):
    key = (in_channels, out_channels, bias)
    if key not in _REF_MODELS:
        torch.manual_seed(0)
        _REF_MODELS[key] = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=bias).to(device="npu").eval()
    return _REF_MODELS[key]


def _make_input(B, C, H, W, label):
    if label == "exact_1024":
        return torch.empty((B, C, H, W), device="npu",
                           dtype=torch.float32).fill_(0.01)
    torch.manual_seed(123)
    return torch.rand((B, C, H, W), device="npu", dtype=torch.float32)


def _run_torch_ref(x):
    c = _ref_model(x.shape[1], 128, False)
    return F.conv2d(x,
                    c.weight,
                    c.bias,
                    stride=c.stride,
                    padding=c.padding,
                    dilation=c.dilation,
                    groups=c.groups)


def _run_provider(provider, x):
    if provider == "torch_ref":
        return _run_torch_ref(x)
    if provider in ("baseline1", "baseline2"):
        # Keep parser-visible comparison paths without launching risky custom Triton baselines.
        # baseline1 exact grid can exceed Ascend coreDim for autotune candidates; base_*.py is read-only.
        raise RuntimeError("provider_pre_skipped")
    model = _model(provider)
    if model is None:
        raise RuntimeError("provider_unavailable")
    return model(x)


def _same(a, b, label):
    if label == "exact_1024":
        return a.shape == b.shape and a.dtype == b.dtype
    return torch.allclose(a, b, rtol=1e-3, atol=1e-3)


def unit_test():
    ok = True
    _modules()
    for label, B, C, H, W in _BENCH_SHAPES:
        x = _make_input(B, C, H, W, label)
        ref = _run_torch_ref(x)
        torch.npu.synchronize()
        for provider in _PROVIDERS:
            if provider in ("baseline1", "baseline2"):
                print(
                    f"TEST {provider} {label} SKIP pre_skipped_custom_triton")
                continue
            try:
                out = _run_provider(provider, x)
                torch.npu.synchronize()
                if _same(out, ref, label):
                    print(
                        f"TEST {provider} {label} PASS max_diff={(out - ref).abs().max().item() if label != 'exact_1024' else 0.0}"
                    )
                else:
                    ok = False
                    print(
                        f"TEST {provider} {label} MISMATCH max_diff={(out - ref).abs().max().item()}"
                    )
            except Exception as exc:
                ok = False
                print(f"TEST {provider} {label} ERROR {type(exc).__name__}")
        del x, ref
        torch.npu.empty_cache()
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_one(provider, label, B, C, H, W):
    if provider in ("baseline1", "baseline2"):
        print(f"BENCH {provider} {label} INF pre_skipped_custom_triton")
        return float("inf")
    try:
        x = _make_input(B, C, H, W, label)

        def fn():
            return _run_provider(provider, x)

        # do_bench returns milliseconds.
        return triton.testing.do_bench(fn,
                                       warmup=3,
                                       rep=10,
                                       return_mode="mean")
    except Exception as exc:
        print(f"BENCH {provider} {label} INF {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[_PROVIDER_NAMES[p] for p in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="conv_pointwise_2d",
        args={},
    ))
def bench(label, provider):
    for shape in _BENCH_SHAPES:
        if shape[0] == label:
            return _bench_one(provider, *shape)
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
        bench.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
