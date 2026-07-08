import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = "3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py"
BASE_FILE = "base_3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py"
OPT_FILE = "opt_3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py"

# label, B, Cin, Cout, D, H, W.  norm_shape is derived from ConvTranspose output W.
_BENCH_SHAPES = [
    ("tiny_triton_path", 1, 4, 8, 4, 8, 8),
    ("medium_acl_path", 2, 8, 16, 8, 16, 16),
    ("default_required", 32, 32, 64, 16, 32, 32),
]

_PROVIDER_FILES = {
    "Baseline Triton1": INPUT_FILE,
    "Baseline Triton2": BASE_FILE,
    "Optimized Triton": OPT_FILE,
}
_MODULES = {}
_MODELS = {}


def _load(filename, key):
    cache_key = (filename, key)
    if cache_key in _MODULES:
        return _MODULES[cache_key]
    path = ROOT / filename
    try:
        spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}",
                                                      path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MODULES[cache_key] = mod
        return mod
    except Exception as exc:
        print(f"INFO provider_unavailable {key}: {type(exc).__name__}")
        _MODULES[cache_key] = None
        return None


class TorchRef(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_transpose(x)
        # Adding a scalar before LayerNorm is mathematically invariant, but keep
        # the source operator semantics in the PyTorch/ACL reference.
        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        return self.gelu(x)


def _out_w(W, kernel=3, stride=2, padding=1, output_padding=1, dilation=1):
    return (W - 1) * stride - 2 * padding + dilation * (kernel -
                                                        1) + output_padding + 1


def _init_args(shape):
    _, B, Cin, Cout, D, H, W = shape
    norm_shape = (_out_w(W), )
    return [
        Cin, Cout, (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), 1.0, norm_shape,
        (2, 2, 2)
    ]


def _device():
    return torch.device("npu")


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _model(provider, shape):
    label = shape[0]
    key = (provider, label)
    if key in _MODELS:
        return _MODELS[key]
    args = _init_args(shape)
    torch.manual_seed(0)
    if provider == "PyTorch / ACL":
        model = TorchRef(*args)
    else:
        mod = _load(_PROVIDER_FILES[provider], provider.replace(" ", "_"))
        if mod is None:
            _MODELS[key] = None
            return None
        model = mod.ModelNew(*args)
    model = model.to(_device()).eval()
    _MODELS[key] = model
    return model


def _make_input(shape):
    label, B, Cin, Cout, D, H, W = shape
    torch.manual_seed(123)
    return torch.rand(B, Cin, D, H, W, device=_device(), dtype=torch.float32)


def _conv_out_numel(shape):
    _, B, Cin, Cout, D, H, W = shape
    Do, Ho, Wo = _out_w(D), _out_w(H), _out_w(W)
    return B * Cout * Do * Ho * Wo


def _should_preskip(provider, shape):
    label = shape[0]
    # The editable baseline computes ROWS_PER_CTA=64 for the default output,
    # yielding 65536 CTAs in both custom epilogues, above Ascend's safe grid cap.
    if provider in ("Baseline Triton1",
                    "Baseline Triton2") and label == "default_required":
        return "comparison_provider_preskipped_to_avoid_grid_cap_poisoning"
    return None


def _run(provider, x, shape):
    reason = _should_preskip(provider, shape)
    if reason:
        raise RuntimeError(reason)
    model = _model(provider, shape)
    if model is None:
        raise RuntimeError("provider_unavailable")
    with torch.no_grad():
        return model(x)


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        x = _make_input(shape)
        ref = _run("PyTorch / ACL", x, shape)
        _sync()
        for provider in [
                "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
        ]:
            reason = _should_preskip(provider, shape)
            if reason:
                print(f"INFO unit_preskip {provider} {shape[0]} {reason}")
                continue
            try:
                y = _run(provider, x, shape)
                _sync()
                max_abs = (y.float() - ref.float()).abs().max().item()
                print(f"CHECK {provider} {shape[0]} max_abs={max_abs:.6g}")
                if provider == "Optimized Triton" and not (max_abs <= 1e-3):
                    ok = False
            except Exception as exc:
                print(
                    f"INFO unit_provider_issue {provider} {shape[0]} {type(exc).__name__}"
                )
                if provider == "Optimized Triton":
                    ok = False
        del x, ref
        gc.collect()
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_provider(provider, shape):
    reason = _should_preskip(provider, shape)
    if reason:
        print(f"INFO bench_preskip {provider} {shape[0]} {reason}")
        return float("inf")
    x = _make_input(shape)
    try:
        # Compile/warmup.
        for _ in range(2):
            _run(provider, x, shape)
        _sync()
        if hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(lambda: _run(provider, x, shape),
                                           warmup=3,
                                           rep=10,
                                           return_mode="mean")
        start = time.perf_counter()
        reps = 10
        for _ in range(reps):
            _run(provider, x, shape)
        _sync()
        return (time.perf_counter() - start) * 1000.0 / reps
    except Exception as exc:
        print(
            f"INFO bench_provider_issue {provider} {shape[0]} {type(exc).__name__}"
        )
        return float("inf")
    finally:
        gc.collect()


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
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="latency_ms",
        plot_name="convtranspose3d_sum_layernorm_avgpool_gelu",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    return _time_provider(provider, shape)


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
        benchmark.run(print_data=True, show_plots=False, save_path=str(ROOT))


if __name__ == "__main__":
    main()
