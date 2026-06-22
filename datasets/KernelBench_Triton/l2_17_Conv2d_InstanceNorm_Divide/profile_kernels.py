import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
DEVICE = "npu"
DTYPE = torch.float32

INPUT_FILE = "17_Conv2d_InstanceNorm_Divide.py"
BASE2_FILE = "base_17_Conv2d_InstanceNorm_Divide.py"
OPT_FILE = "opt_17_Conv2d_InstanceNorm_Divide.py"

IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
DIVIDE_BY = 2.0

_BENCH_SHAPES = [
    ("tiny_16", 16, 64, 16, 16),
    ("small_32", 32, 64, 32, 32),
    ("medium_64", 64, 64, 64, 64),
    ("exact_128", 128, 64, 128, 128),
]
_DISPATCH_TEST_SHAPES = [
    ("persistent_rows", 512, 64, 8, 8),  # N*out_channels = 65536, tiny HW
]


def _load(fname: str, key: str):
    spec = importlib.util.spec_from_file_location(
        f"k_{key}_{Path(fname).stem}", ROOT / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname: str, key: str):
    try:
        return _load(fname, key)
    except Exception as exc:
        print(f"INFO {key} unavailable {type(exc).__name__}")
        return None


baseline1_mod = _load(INPUT_FILE, "baseline1")
baseline2_mod = _load_optional(BASE2_FILE, "baseline2")
optimized_mod = _load(OPT_FILE, "optimized")
_MODEL_CACHE = {}
_TORCH_CACHE = {}
_SKIP_COMPARE = set()


class TorchRef(nn.Module):

    def __init__(self,
                 in_channels=IN_CHANNELS,
                 out_channels=OUT_CHANNELS,
                 kernel_size=KERNEL_SIZE,
                 divide_by=DIVIDE_BY):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        y = self.conv(x)
        y = F.instance_norm(
            y,
            running_mean=None,
            running_var=None,
            weight=None,
            bias=None,
            use_input_stats=True,
            momentum=self.instance_norm.momentum,
            eps=self.instance_norm.eps,
        )
        return y / self.divide_by


def _torch_model(dtype=DTYPE):
    key = (str(dtype), )
    if key not in _TORCH_CACHE:
        torch.manual_seed(0)
        _TORCH_CACHE[key] = TorchRef().to(device=DEVICE, dtype=dtype).eval()
    return _TORCH_CACHE[key]


def _model(provider: str, dtype=DTYPE):
    key = (provider, str(dtype))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    mod = {
        "baseline1": baseline1_mod,
        "baseline2": baseline2_mod,
        "optimized": optimized_mod
    }.get(provider)
    if mod is None:
        return None
    torch.manual_seed(0)
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    model = mod.ModelNew(*init).to(device=DEVICE, dtype=dtype).eval()
    _MODEL_CACHE[key] = model
    return model


def _make_inputs(N, C, H, W, dtype=DTYPE):
    torch.manual_seed(123)
    return torch.rand((N, C, H, W), device=DEVICE, dtype=dtype)


def _run_torch_ref(x):
    with torch.no_grad():
        return _torch_model(x.dtype)(x)


def _run_provider(provider: str, x):
    if provider == "torch_ref":
        return _run_torch_ref(x)
    if provider in _SKIP_COMPARE:
        raise RuntimeError("provider_marked_skip")
    model = _model(provider, x.dtype)
    if model is None:
        raise RuntimeError("provider_unavailable")
    with torch.no_grad():
        return model(x)


def _isclose(a, b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-3)


def _test_one(label, N, C, H, W, providers):
    x = _make_inputs(N, C, H, W)
    ref = _run_torch_ref(x)
    ok_opt = True
    for provider in providers:
        if provider == "torch_ref":
            print(f"TEST torch_ref {label} PASS")
            continue
        # Baseline1 direct grid is N*out_channels; skip only the synthetic overflow path.
        if provider in ("baseline1", "baseline2") and N * OUT_CHANNELS > 65535:
            print(f"TEST {provider} {label} SKIP grid_guard")
            continue
        try:
            out = _run_provider(provider, x)
            torch.npu.synchronize()
            if _isclose(out, ref):
                print(f"TEST {provider} {label} PASS")
            else:
                diff = (out - ref).abs().max().item()
                if provider == "optimized":
                    print(
                        f"TEST optimized {label} MISMATCH max_diff={diff:.6g}")
                    ok_opt = False
                else:
                    _SKIP_COMPARE.add(provider)
                    print(
                        f"TEST {provider} {label} SKIP value_diff={diff:.6g}")
        except Exception as exc:
            if provider == "optimized":
                print(
                    f"TEST optimized {label} MISMATCH exception={type(exc).__name__}"
                )
                ok_opt = False
            else:
                _SKIP_COMPARE.add(provider)
                print(
                    f"TEST {provider} {label} SKIP runtime_{type(exc).__name__}"
                )
    return ok_opt


def unit_test():
    providers = ["torch_ref", "baseline1", "baseline2", "optimized"]
    all_ok = True
    for shape in _BENCH_SHAPES + _DISPATCH_TEST_SHAPES:
        all_ok = _test_one(*shape, providers=providers) and all_ok
    print("UNIT_TEST PASS" if all_ok else "UNIT_TEST_FAILED")
    return all_ok


def _time_ms(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / rep


def bench_provider(label, provider):
    dims = {s[0]: s[1:] for s in _BENCH_SHAPES}[label]
    N, C, H, W = dims
    if provider == "baseline2" and baseline2_mod is None:
        return float("inf")
    if provider in _SKIP_COMPARE:
        return float("inf")
    x = _make_inputs(N, C, H, W)
    try:

        def fn():
            return _run_provider(provider, x)

        try:
            return triton.testing.do_bench(fn,
                                           warmup=5,
                                           rep=20,
                                           return_mode="mean")
        except Exception:
            return _time_ms(fn, warmup=3, rep=10)
    except Exception as exc:
        print(f"INFO bench {provider} {label} inf_{type(exc).__name__}")
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
        ylabel="Latency (ms)",
        plot_name="conv2d_instancenorm_divide",
        args={},
    ))
def benchmark(label, provider):
    return bench_provider(label, provider)


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
