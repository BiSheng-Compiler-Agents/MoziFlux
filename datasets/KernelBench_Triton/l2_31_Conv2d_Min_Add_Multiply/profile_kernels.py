import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

ROOT = Path(__file__).resolve().parent
_INIT_ARGS = [64, 128, 3, 0.5, (128, 1, 1), 2.0]
_BENCH_SHAPES = [
    ("small_32", 4, 64, 32, 32),
    ("medium_64", 16, 64, 64, 64),
    ("exact_128", 128, 64, 128, 128),
]
_SHAPE_BY_LABEL = {row[0]: row[1:] for row in _BENCH_SHAPES}
_PROVIDERS = ["torch_ref", "baseline1", "baseline2", "optimized"]
_PROVIDER_NAMES = {
    "torch_ref": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}
_MODEL_CACHE = {}
_BASELINE2_AVAILABLE = False  # base_*.py is read-only reference and intentionally not loaded.


def _load(fname: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


baseline1_mod = _load("31_Conv2d_Min_Add_Multiply.py",
                      "k_baseline1_31_conv2d_min_add_multiply")
opt_mod = _load("opt_31_Conv2d_Min_Add_Multiply.py",
                "k_optimized_31_conv2d_min_add_multiply")


class RefModel(nn.Module):

    def __init__(
            self,
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            constant_value=0.5,
            bias_shape=(128, 1, 1),
            scaling_factor=2.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = float(constant_value)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        y = self.conv(x)
        return (torch.minimum(
            y, torch.tensor(
                self.constant_value, device=y.device, dtype=y.dtype)) +
                self.bias.to(device=y.device,
                             dtype=y.dtype)) * self.scaling_factor


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _make_inputs(shape):
    torch.manual_seed(123)
    return (torch.rand(shape, device="npu"), )


def _model(provider: str, dtype=torch.float32):
    key = (provider, dtype)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    if provider == "torch_ref":
        model = RefModel(*_INIT_ARGS)
    elif provider == "baseline1":
        model = baseline1_mod.ModelNew(*_INIT_ARGS)
    elif provider == "optimized":
        model = opt_mod.ModelNew(*_INIT_ARGS)
    else:
        return None
    model = model.to(device="npu", dtype=dtype).eval()
    _MODEL_CACHE[key] = model
    return model


def _run_provider(provider: str, x):
    if provider == "baseline2":
        raise RuntimeError("baseline2_readonly_not_loaded")
    with torch.no_grad():
        return _model(provider, x.dtype)(x)


def _max_diff(a, b):
    return (a - b).abs().max().item()


def unit_test():
    ok = True
    for label, b, c, h, w in _BENCH_SHAPES:
        x, = _make_inputs((b, c, h, w))
        ref = _run_provider("torch_ref", x)
        _sync()
        for provider in ["baseline1", "baseline2", "optimized"]:
            if provider == "baseline2" and not _BASELINE2_AVAILABLE:
                print(
                    f"TEST {provider} {label} SKIP baseline2_readonly_not_loaded"
                )
                continue
            try:
                y = _run_provider(provider, x)
                _sync()
                diff = _max_diff(y, ref)
                if torch.allclose(y, ref, rtol=1e-3, atol=1e-3):
                    print(f"TEST {provider} {label} PASS max_diff={diff:.6g}")
                else:
                    print(
                        f"TEST {provider} {label} MISMATCH max_diff={diff:.6g}"
                    )
                    if provider == "optimized":
                        ok = False
            except Exception as exc:
                token = "SKIP" if provider != "optimized" else "FAIL"
                print(f"TEST {provider} {label} {token} {type(exc).__name__}")
                if provider == "optimized":
                    ok = False
        del x, ref
        gc.collect()
    # Unit-test-only persistent dispatch coverage: many planes, tiny spatial size.
    try:
        x, = _make_inputs((1024, 64, 5, 5))
        ref = _run_provider("torch_ref", x)
        y = _run_provider("optimized", x)
        _sync()
        diff = _max_diff(y, ref)
        if torch.allclose(y, ref, rtol=1e-3, atol=1e-3):
            print(
                f"TEST optimized persistent_dispatch PASS max_diff={diff:.6g}")
        else:
            print(
                f"TEST optimized persistent_dispatch MISMATCH max_diff={diff:.6g}"
            )
            ok = False
    except Exception as exc:
        print(f"TEST optimized persistent_dispatch FAIL {type(exc).__name__}")
        ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_ms(fn, warmup=5, rep=20):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
            _sync()
        start = time.perf_counter()
        for _ in range(rep):
            fn()
            _sync()
        end = time.perf_counter()
        return (end - start) * 1000.0 / rep


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[row[0] for row in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[_PROVIDER_NAMES[p] for p in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="conv2d_min_add_multiply",
        args={},
    ))
def benchmark(label, provider):
    if provider == "baseline2" and not _BASELINE2_AVAILABLE:
        print(
            f"INFO benchmark {provider} {label} baseline2_readonly_not_loaded")
        return float("inf")
    shape = _SHAPE_BY_LABEL[label]
    try:
        x, = _make_inputs(shape)

        def fn():
            return _run_provider(provider, x)

        return _time_ms(fn)
    except Exception as exc:
        print(
            f"INFO benchmark {provider} {label} unavailable {type(exc).__name__}"
        )
        return float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not args.test and not args.bench:
        args.test = True
        args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
