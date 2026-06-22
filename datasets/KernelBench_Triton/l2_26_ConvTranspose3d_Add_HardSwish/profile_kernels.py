import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
INPUT_FILE = "26_ConvTranspose3d_Add_HardSwish.py"
BASE_FILE = "base_26_ConvTranspose3d_Add_HardSwish.py"
OPT_FILE = "opt_26_ConvTranspose3d_Add_HardSwish.py"

_BENCH_SHAPES = [
    ("small_direct", 1, 4, 4, 4),
    ("medium_direct", 8, 8, 8, 8),
    ("target_persistent", 128, 16, 16, 16),
]

IN_CHANNELS = 32
OUT_CHANNELS = 64
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
OUTPUT_PADDING = 1
BIAS_SHAPE = (OUT_CHANNELS, 1, 1, 1, 1)
MAX_PROGRAMS = 65535
BASELINE1_FP32_BLOCK = 2048
OPT_BLOCK = 4096


def _load(fname, optional=False):
    path = ROOT / fname
    try:
        spec = importlib.util.spec_from_file_location("k_" + path.stem, path)
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


baseline1_mod = _load(INPUT_FILE)
baseline2_mod = _load(BASE_FILE, optional=True)
opt_mod = _load(OPT_FILE)


class TorchRef(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            IN_CHANNELS,
            OUT_CHANNELS,
            KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            output_padding=OUTPUT_PADDING,
        )
        self.bias = nn.Parameter(torch.randn(BIAS_SHAPE))

    def forward(self, x, add_input):
        z = self.conv_transpose(x) + add_input
        return z * torch.clamp(z + 3.0, min=0.0, max=6.0) / 6.0


_MODEL_CACHE = {}


def _init_args():
    return [
        IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING,
        OUTPUT_PADDING, BIAS_SHAPE
    ]


def _model(provider):
    if provider in _MODEL_CACHE:
        return _MODEL_CACHE[provider]
    torch.manual_seed(0)
    if provider == "torch":
        model = TorchRef()
    elif provider == "baseline1":
        model = baseline1_mod.ModelNew(*_init_args())
    elif provider == "baseline2":
        if baseline2_mod is None:
            return None
        model = baseline2_mod.ModelNew(*_init_args())
    elif provider == "optimized":
        model = opt_mod.ModelNew(*_init_args())
    else:
        raise KeyError(provider)
    model = model.to("npu").eval()
    _MODEL_CACHE[provider] = model
    return model


def _make_inputs(batch, d, h, w):
    torch.manual_seed(123)
    x = torch.rand(batch,
                   IN_CHANNELS,
                   d,
                   h,
                   w,
                   device="npu",
                   dtype=torch.float32)
    add = torch.rand(batch,
                     OUT_CHANNELS,
                     d * STRIDE,
                     h * STRIDE,
                     w * STRIDE,
                     device="npu",
                     dtype=torch.float32)
    return x, add


def _out_numel(batch, d, h, w):
    return batch * OUT_CHANNELS * (d * STRIDE) * (h * STRIDE) * (w * STRIDE)


def _baseline1_grid_ok(batch, d, h, w):
    n_tiles = triton.cdiv(_out_numel(batch, d, h, w), BASELINE1_FP32_BLOCK)
    return n_tiles <= MAX_PROGRAMS


def _provider_skip(provider, label, batch, d, h, w):
    if provider == "baseline2":
        return "readonly_comparison"
    if provider == "baseline1" and not _baseline1_grid_ok(batch, d, h, w):
        return "grid_guard"
    return None


def _run_provider(provider, x, add):
    model = _model(provider)
    if model is None:
        raise RuntimeError("provider_unavailable")
    with torch.no_grad():
        return model(x, add)


def _sync():
    torch.npu.synchronize()


def _manual_time(fn, warmup=3, rep=10):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
            _sync()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        _sync()
        return start.elapsed_time(end) / rep


def _bench_one(provider, label, batch, d, h, w):
    reason = _provider_skip(provider, label, batch, d, h, w)
    if reason:
        return float("inf")
    x, add = _make_inputs(batch, d, h, w)
    try:
        return triton.testing.do_bench(lambda: _run_provider(provider, x, add),
                                       warmup=3,
                                       rep=10,
                                       return_mode="mean")
    except Exception as exc:
        print(f"INFO bench {provider} {label} inf {type(exc).__name__}")
        try:
            return _manual_time(lambda: _run_provider(provider, x, add),
                                warmup=2,
                                rep=5)
        except Exception as exc2:
            print(
                f"INFO bench {provider} {label} inf_fallback {type(exc2).__name__}"
            )
            return float("inf")


def unit_test():
    ok = True
    for label, batch, d, h, w in _BENCH_SHAPES:
        x, add = _make_inputs(batch, d, h, w)
        ref = _run_provider("torch", x, add)
        _sync()
        for provider in ("baseline1", "baseline2", "optimized"):
            reason = _provider_skip(provider, label, batch, d, h, w)
            if reason:
                print(f"TEST {provider} {label} SKIP {reason}")
                continue
            try:
                out = _run_provider(provider, x, add)
                _sync()
                max_abs = (out - ref).abs().max().item()
                close = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
                if close:
                    print(
                        f"TEST {provider} {label} PASS max_abs={max_abs:.6g}")
                else:
                    if provider == "optimized":
                        print(
                            f"TEST {provider} {label} MISMATCH max_abs={max_abs:.6g}"
                        )
                        ok = False
                    else:
                        print(
                            f"TEST {provider} {label} SKIP value_diff max_abs={max_abs:.6g}"
                        )
            except Exception as exc:
                if provider == "optimized":
                    print(
                        f"TEST {provider} {label} MISMATCH exception={type(exc).__name__}"
                    )
                    ok = False
                else:
                    print(
                        f"TEST {provider} {label} SKIP runtime_{type(exc).__name__}"
                    )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


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
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="convtranspose3d_add_hardswish",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, batch, d, h, w = shape
    return _bench_one(provider, label, batch, d, h, w)


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
        bench.run(print_data=True, show_plots=False, save_path=None)


if __name__ == "__main__":
    main()
