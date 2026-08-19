import argparse
import importlib.util
import pathlib
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

ROOT = pathlib.Path(__file__).resolve().parent
BASELINE1_FILE = "38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply.py"
BASELINE2_FILE = "base_38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply.py"
OPT_FILE = "opt_38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply.py"

_BENCH_SHAPES = [
    ("direct_small", 2, 32, 4, 8, 8),
    ("direct_medium", 4, 32, 8, 16, 16),
    ("target_acl", 32, 32, 32, 64, 64),
]

_PROVIDER_FILES = {
    "Baseline Triton1": BASELINE1_FILE,
    "Baseline Triton2": BASELINE2_FILE,
    "Optimized Triton": OPT_FILE,
}
_PROVIDER_ORDER = [
    "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
]
_MODULES: Dict[str, object] = {}
_MODELS: Dict[str, nn.Module] = {}


def _load(path_name: str, key: str):
    if key in _MODULES:
        return _MODULES[key]
    path = ROOT / path_name
    spec = importlib.util.spec_from_file_location(f"k38_{key}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULES[key] = mod
    return mod


class TorchRef(nn.Module):

    def __init__(self,
                 in_channels=32,
                 out_channels=64,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 output_padding=1,
                 pool_kernel_size=2,
                 clamp_min=0.0,
                 clamp_max=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 output_padding=output_padding)
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.avg_pool(x)
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return torch.softmax(x, dim=1) * 2.0


def _init_args():
    return [32, 64, 3, 2, 1, 1, 2, 0.0, 1.0]


def _model(provider: str) -> nn.Module:
    if provider in _MODELS:
        return _MODELS[provider]
    torch.manual_seed(0)
    if provider == "PyTorch / ACL":
        model = TorchRef(*_init_args())
    else:
        mod = _load(_PROVIDER_FILES[provider], provider.replace(" ", "_"))
        model = mod.ModelNew(*_init_args())
    model = model.eval().npu()
    _MODELS[provider] = model
    return model


def _make_input(shape: Tuple) -> torch.Tensor:
    _, N, C, D, H, W = shape
    torch.manual_seed(123)
    return torch.rand((N, C, D, H, W), device="npu", dtype=torch.float32)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _run_provider(provider: str, x: torch.Tensor):
    with torch.no_grad():
        return _model(provider)(x)


def _max_diff(a, b):
    d = (a - b).abs()
    return float(d.max().detach().cpu()) if d.numel() else 0.0


def unit_test() -> bool:
    ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_input(shape)
        ref = _run_provider("PyTorch / ACL", x)
        _sync()
        for provider in _PROVIDER_ORDER[1:]:
            try:
                out = _run_provider(provider, x)
                _sync()
                md = _max_diff(out, ref)
                if md > 1e-3:
                    raise AssertionError(f"max_diff={md:.6g}")
                print(f"OK {provider} {label} max_diff={md:.6g}")
            except Exception as exc:
                if provider == "Optimized Triton":
                    ok = False
                    detail = str(exc).replace("\n", " ")[:180]
                    print(
                        f"UNIT_TEST_FAILED Optimized Triton {label}: {type(exc).__name__} {detail}"
                    )
                else:
                    print(
                        f"INFO comparison_provider_unavailable {provider} {label}: {type(exc).__name__}"
                    )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_once(provider: str, shape: Tuple) -> float:
    x = _make_input(shape)
    try:
        # Guard known baseline1 default-shape grid poison: N*cdiv(DHW,64)=65536.
        if provider == "Baseline Triton1" and shape[0] == "target_acl":
            print(
                "INFO comparison_provider_preskipped Baseline Triton1 target_acl grid_cap"
            )
            return float("inf")
        _run_provider(provider, x)
        _sync()

        def fn():
            _run_provider(provider, x)

        try:
            return triton.testing.do_bench(fn,
                                           warmup=10,
                                           rep=50,
                                           return_mode="mean")
        except Exception:
            times = []
            for _ in range(5):
                fn()
                _sync()
            for _ in range(20):
                t0 = time.perf_counter()
                fn()
                _sync()
                times.append((time.perf_counter() - t0) * 1000.0)
            return sum(times) / len(times)
    except Exception as exc:
        print(
            f"INFO bench_unavailable {provider} {shape[0]}: {type(exc).__name__}"
        )
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDER_ORDER,
        line_names=_PROVIDER_ORDER,
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="convtranspose3d_avgpool_clamp_softmax_multiply",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    return _bench_once(provider, shape)


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
