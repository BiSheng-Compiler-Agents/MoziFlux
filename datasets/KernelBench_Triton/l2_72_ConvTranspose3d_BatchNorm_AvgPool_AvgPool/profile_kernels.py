import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool.py"
OPT_FILE = ROOT / "opt_72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool.py"
BASE2_PRESENT_BUT_SANDBOXED = True
_MAX_GRID = 65535

_BENCH_SHAPES = [
    ("tiny_direct", 1, 3, 8, 8, 8),
    ("medium_direct", 4, 3, 16, 16, 16),
    ("default_required", 64, 3, 32, 32, 32),
]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod_cache = {}
_model_cache = {}


def _module(key):
    if key not in _mod_cache:
        if key == "baseline1":
            _mod_cache[key] = _load(INPUT_FILE, "k_l2_72_baseline1")
        elif key == "optimized":
            _mod_cache[key] = _load(OPT_FILE, "k_l2_72_optimized")
        else:
            raise KeyError(key)
    return _mod_cache[key]


class TorchRef(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=16,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 bias_shape=(16, 1, 1, 1)):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.bias_shape = bias_shape

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = F.avg_pool3d(x, kernel_size=2, stride=2)
        return F.avg_pool3d(x, kernel_size=2, stride=2)


def _device():
    return torch.device("npu")


def _model(key):
    if key in _model_cache:
        return _model_cache[key]
    torch.manual_seed(0)
    if key == "torch_ref":
        m = TorchRef()
    else:
        mod = _module(key)
        init = mod.get_init_inputs()
        if init == [()]:
            init = []
        m = mod.ModelNew(*init)
    m = m.to(_device())
    _model_cache[key] = m
    return m


def _make_input(label, batch, channels, depth, height, width):
    torch.manual_seed(1234 + batch + depth + height + width)
    return torch.rand(batch, channels, depth, height, width, device=_device())


def _baseline1_grid_overflows(batch, depth, height, width):
    conv_d = (depth - 1) * 2 - 2 * 1 + 3
    conv_h = (height - 1) * 2 - 2 * 1 + 3
    conv_w = (width - 1) * 2 - 2 * 1 + 3
    od, oh, ow = conv_d // 4, conv_h // 4, conv_w // 4
    axis0 = batch * 16 * od * oh
    return axis0 * triton.cdiv(ow, 32) > _MAX_GRID


def _run_provider(provider, x, shape):
    label, batch, channels, depth, height, width = shape
    if provider == "PyTorch / ACL":
        return _model("torch_ref")(x)
    if provider == "Baseline Triton1":
        if _baseline1_grid_overflows(batch, depth, height, width):
            raise RuntimeError("comparison_provider_preskipped_grid_overflow")
        return _model("baseline1")(x)
    if provider == "Baseline Triton2":
        raise RuntimeError("sandboxed_reference_base_file_not_read")
    if provider == "Optimized Triton":
        return _model("optimized")(x)
    raise KeyError(provider)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_input(*shape)
        ref = _run_provider("PyTorch / ACL", x, shape)
        for provider in [
                "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
        ]:
            try:
                y = _run_provider(provider, x, shape)
                diff = _max_abs(y, ref)
                passed = math.isfinite(diff) and diff <= 1e-3
                print(
                    f"TEST {provider} {label}: {'PASS' if passed else 'MISMATCH'} max_abs={diff:.6g}"
                )
                if provider == "Optimized Triton" and not passed:
                    ok = False
            except Exception as exc:
                tag = type(exc).__name__
                if provider == "Optimized Triton":
                    print(
                        f"TEST {provider} {label}: EXCEPTION {tag} max_abs=inf"
                    )
                    ok = False
                else:
                    print(
                        f"TEST {provider} {label}: SKIP_UNAVAILABLE {tag} max_abs=inf"
                    )

    # Forced Triton fallback coverage without huge allocation: disable production ACL dispatch.
    for forced_name, forced_grid in [("forced_direct_triton", _MAX_GRID),
                                     ("forced_persistent_triton", 1)]:
        try:
            opt_mod = _module("optimized")
            old_grid = getattr(opt_mod, "_MAX_GRID", None)
            old_acl = getattr(opt_mod, "_USE_ACL_DISPATCH", None)
            opt_mod._USE_ACL_DISPATCH = False
            opt_mod._MAX_GRID = forced_grid
            _model_cache.pop("optimized", None)
            shape = (forced_name, 1, 3, 16, 16, 16)
            x = _make_input(*shape)
            ref = _run_provider("PyTorch / ACL", x, shape)
            y = _run_provider("Optimized Triton", x, shape)
            diff = _max_abs(y, ref)
            passed = math.isfinite(diff) and diff <= 1e-3
            print(
                f"TEST Optimized Triton {forced_name}: {'PASS' if passed else 'MISMATCH'} max_abs={diff:.6g}"
            )
            ok = ok and passed
            if old_grid is not None:
                opt_mod._MAX_GRID = old_grid
            if old_acl is not None:
                opt_mod._USE_ACL_DISPATCH = old_acl
            _model_cache.pop("optimized", None)
        except Exception as exc:
            print(
                f"TEST Optimized Triton {forced_name}: EXCEPTION {type(exc).__name__} max_abs=inf"
            )
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
        for _ in range(2):
            fn()
            torch.npu.synchronize()
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / reps


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="l2_72_convtranspose3d_bn_avgpool_avgpool",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = _make_input(*shape)
    try:
        if provider == "Baseline Triton1" and _baseline1_grid_overflows(
                shape[1], shape[3], shape[4], shape[5]):
            print(
                f"INFO {provider} {label}: inf comparison_provider_preskipped_grid_overflow"
            )
            return float("inf")
        if provider == "Baseline Triton2":
            print(
                f"INFO {provider} {label}: inf sandboxed_reference_base_file_not_read"
            )
            return float("inf")
        _run_provider(provider, x, shape)
        torch.npu.synchronize()
        return _time_ms(lambda: _run_provider(provider, x, shape))
    except Exception as exc:
        print(f"INFO {provider} {label}: inf {type(exc).__name__}")
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
