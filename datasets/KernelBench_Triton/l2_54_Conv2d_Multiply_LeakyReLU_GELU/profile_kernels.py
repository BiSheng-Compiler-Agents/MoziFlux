import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

THIS_DIR = Path(__file__).resolve().parent
INPUT_FILE = "54_Conv2d_Multiply_LeakyReLU_GELU.py"
BASE_FILE = "base_54_Conv2d_Multiply_LeakyReLU_GELU.py"
OPT_FILE = "opt_54_Conv2d_Multiply_LeakyReLU_GELU.py"

_BENCH_SHAPES = [
    ("small_2x64x64", 2, 64, 64, 64, 3),
    ("irregular_3x64x70", 3, 64, 70, 67, 3),
    ("target_64x256", 64, 64, 256, 256, 3),
]

_MODULES = {}
_MODELS = {}


def _load(filename, key):
    if key in _MODULES:
        return _MODULES[key]
    path = THIS_DIR / filename
    spec = importlib.util.spec_from_file_location(f"k_l2_54_{key}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULES[key] = mod
    return mod


class TorchRef(nn.Module):

    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=3,
                 multiplier_shape=(64, 1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier.to(device=x.device, dtype=x.dtype)
        x = self.leaky_relu(x)
        return F.gelu(x)


def _make_inputs(label, batch, channels, height, width, kernel_size):
    torch.manual_seed(123)
    x = torch.rand(batch,
                   channels,
                   height,
                   width,
                   device="npu",
                   dtype=torch.float32)
    return (x, )


def _model(key, shape):
    _, _, channels, _, _, kernel_size = shape
    init = [channels, 64, kernel_size, (64, 1, 1)]
    cache_key = (key, tuple(init))
    if cache_key in _MODELS:
        return _MODELS[cache_key]
    torch.manual_seed(0)
    if key == "torch_ref":
        model = TorchRef(*init)
    else:
        filename = {
            "baseline1": INPUT_FILE,
            "baseline2": BASE_FILE,
            "optimized": OPT_FILE
        }[key]
        mod = _load(filename, key)
        model = mod.ModelNew(*init)
    model = model.to(device="npu", dtype=torch.float32).eval()
    _MODELS[cache_key] = model
    return model


def _run_torch_ref(inputs, shape):
    with torch.no_grad():
        return _model("torch_ref", shape)(*inputs)


def _run_provider(key, inputs, shape):
    with torch.no_grad():
        return _model(key, shape)(*inputs)


def _max_diff(a, b):
    return (a - b).abs().max().item()


def _allclose(a, b, tol=1e-3):
    return _max_diff(a, b) <= tol


def _force_optimized_persistent(inputs, shape):
    mod = _load(OPT_FILE, "optimized")
    old = getattr(mod, "_MAX_GRID", None)
    if old is None:
        return None
    mod._MAX_GRID = 1
    try:
        _MODELS.clear()
        return _run_provider("optimized", inputs, shape)
    finally:
        mod._MAX_GRID = old
        _MODELS.clear()


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        inputs = _make_inputs(*shape)
        ref = _run_torch_ref(inputs, shape)
        for key, name in [("baseline1", "Baseline Triton1"),
                          ("baseline2", "Baseline Triton2"),
                          ("optimized", "Optimized Triton")]:
            try:
                out = _run_provider(key, inputs, shape)
                diff = _max_diff(out, ref)
                close = diff <= 1e-3
                print(
                    f"UNIT {label} {name} max_abs_diff={diff:.6g} close={close}"
                )
                if key == "optimized" and not close:
                    ok = False
            except BaseException as e:
                et = type(e).__name__
                print(f"INFO {label} {name} unavailable_or_preskipped {et}")
                if key == "optimized":
                    ok = False
        if label == "small_2x64x64":
            try:
                pout = _force_optimized_persistent(inputs, shape)
                if pout is not None:
                    diff = _max_diff(pout, ref)
                    close = diff <= 1e-3
                    print(
                        f"UNIT {label} Optimized Triton forced_persistent max_abs_diff={diff:.6g} close={close}"
                    )
                    ok = ok and close
            except BaseException as e:
                print(
                    f"INFO {label} Optimized Triton forced_persistent unavailable {type(e).__name__}"
                )
                ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _manual_bench(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / rep


def bench_one(provider, label):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    inputs = _make_inputs(*shape)
    if provider == "torch_ref":

        def fn():
            return _run_torch_ref(inputs, shape)
    else:
        key = {
            "baseline1": "baseline1",
            "baseline2": "baseline2",
            "optimized": "optimized"
        }[provider]

        def fn():
            return _run_provider(key, inputs, shape)

    try:
        ms = triton.testing.do_bench(fn, warmup=10, rep=30, return_mode="mean")
        return ms
    except BaseException:
        try:
            return _manual_bench(fn)
        except BaseException as e:
            print(
                f"INFO bench {label} {provider} unavailable_or_preskipped {type(e).__name__}"
            )
            return float("inf")


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
        styles=[("blue", "-"), ("green", "-"), ("black", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="l2_54_conv2d_multiply_leakyrelu_gelu",
        args={},
    ))
def benchmark(label, provider):
    return bench_one(provider, label)


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
        benchmark.run(print_data=True,
                      show_plots=False,
                      save_path=str(THIS_DIR))


if __name__ == "__main__":
    main()
