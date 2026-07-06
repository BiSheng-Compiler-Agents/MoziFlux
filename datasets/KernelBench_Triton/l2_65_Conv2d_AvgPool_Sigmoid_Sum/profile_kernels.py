import argparse
import importlib.util
import math
import pathlib
import sys
import time

import torch
import torch_npu  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
import triton

ROOT = pathlib.Path(__file__).resolve().parent
INPUT_FILE = ROOT / "65_Conv2d_AvgPool_Sigmoid_Sum.py"
OPT_FILE = ROOT / "opt_65_Conv2d_AvgPool_Sigmoid_Sum.py"
# The sandbox marks base_*.py as DO NOT read.  Keep parser-visible Baseline Triton2
# records/column, but intentionally do not import the read-only reference file.
BASE2_AVAILABLE = False

_BENCH_SHAPES = [
    ("small", 4, 8, 64, 64, 64, 3, 4),
    ("irregular", 3, 8, 71, 67, 64, 3, 4),
    ("target", 128, 8, 384, 384, 64, 3, 4),
]

_MODEL_CACHE = {}
_REF_CACHE = {}


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mods = {}


def _module(key):
    if key not in _mods:
        if key == "baseline1":
            _mods[key] = _load(INPUT_FILE, "k65_baseline1")
        elif key == "opt":
            _mods[key] = _load(OPT_FILE, "k65_opt")
        else:
            raise KeyError(key)
    return _mods[key]


class TorchRef(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        y = self.conv(x)
        y = self.avg_pool(y)
        y = torch.sigmoid(y)
        return torch.sum(y, dim=(1, 2, 3))


def _init_args(shape):
    label, B, in_ch, H, W, out_ch, k, pool_k = shape
    return [in_ch, out_ch, k, pool_k]


def _model(key, shape):
    label = shape[0]
    cache_key = (key, label)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    torch.manual_seed(0)
    if key == "ref":
        model = TorchRef(*_init_args(shape))
    elif key in ("baseline1", "opt"):
        model = _module(key).ModelNew(*_init_args(shape))
    else:
        raise KeyError(key)
    model = model.eval().npu()
    _MODEL_CACHE[cache_key] = model
    return model


def _make_inputs(shape):
    label, B, in_ch, H, W, out_ch, k, pool_k = shape
    torch.manual_seed(123)
    x = torch.rand(B, in_ch, H, W, device="npu", dtype=torch.float32)
    return [x]


def _run_torch_ref(shape, inputs):
    return _model("ref", shape)(*inputs)


def _run_provider(key, shape, inputs):
    if key == "baseline2":
        raise RuntimeError("sandbox_base_file_read_disallowed")
    return _model(key, shape)(*inputs)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        inputs = _make_inputs(shape)
        with torch.no_grad():
            ref = _run_torch_ref(shape, inputs)
            for key, display in [
                ("baseline1", "Baseline Triton1"),
                ("baseline2", "Baseline Triton2"),
                ("opt", "Optimized Triton"),
            ]:
                if key == "baseline2":
                    print(f"TEST {display} {label}: SKIP_UNAVAILABLE sandbox_base_file_read_disallowed max_abs=inf")
                    continue
                if key == "baseline1" and label == "target":
                    print(f"TEST {display} {label}: SKIP_COMPARISON target_custom_triton_too_slow max_abs=inf")
                    continue
                try:
                    out = _run_provider(key, shape, inputs)
                    diff = _max_abs(out, ref)
                    passed = diff <= 1e-3
                    print(f"TEST {display} {label}: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}")
                    if key == "opt" and not passed:
                        ok = False
                except Exception as exc:
                    tag = type(exc).__name__
                    if key == "opt":
                        print(f"TEST {display} {label}: FAIL_EXCEPTION {tag} max_abs=inf")
                        ok = False
                    else:
                        print(f"TEST {display} {label}: SKIP_UNAVAILABLE {tag} max_abs=inf")
        torch.npu.synchronize()
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_one(provider, shape):
    label = shape[0]
    if provider == "Baseline Triton2":
        print(f"INFO bench_skip {provider} {label}: sandbox_base_file_read_disallowed")
        return float("inf")
    key = {"PyTorch / ACL": "ref", "Baseline Triton1": "baseline1", "Optimized Triton": "opt"}[provider]
    if key == "baseline1" and label == "target":
        print(f"INFO bench_skip {provider} {label}: target_custom_triton_too_slow")
        return float("inf")
    inputs = _make_inputs(shape)
    try:
        # Warm-up first; if triton.testing.do_bench is unavailable/unreliable, use perf_counter.
        fn = lambda: (_run_torch_ref(shape, inputs) if key == "ref" else _run_provider(key, shape, inputs))
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        if hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn, warmup=10, rep=30, return_mode="mean")
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return sum(times) / len(times)
    except Exception as exc:
        print(f"INFO bench_unavailable {provider} {label}: {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="conv2d_avgpool_sigmoid_sum",
        args={},
    )
)
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    return _bench_one(provider, shape)


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
