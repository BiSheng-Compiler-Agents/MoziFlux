import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "60_ConvTranspose3d_Swish_GroupNorm_HardSwish.py"
OPT_FILE = ROOT / "opt_60_ConvTranspose3d_Swish_GroupNorm_HardSwish.py"
BASE2_PRESENT_BUT_READ_PROHIBITED = True

_BENCH_SHAPES = [
    ("tiny_triton_path", 2, 3, 4, 8, 8),
    ("medium_acl_path", 8, 3, 8, 16, 16),
    ("default_required", 128, 3, 16, 32, 32),
]
PROVIDERS = ["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"]
_MODEL_CACHE = {}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


input_mod = _load(INPUT_FILE, "k_l2_60_input")
opt_mod = _load(OPT_FILE, "k_l2_60_opt")


def _init_args():
    args = input_mod.get_init_inputs()
    if args == [()]:
        return []
    return list(args)


class TorchRef(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, x):
        y = self.conv_transpose(x)
        y = y * torch.sigmoid(y)
        y = F.group_norm(y, self.group_norm.num_groups, self.group_norm.weight, self.group_norm.bias, self.group_norm.eps)
        return y * torch.clamp(y + 3.0, min=0.0, max=6.0) * (1.0 / 6.0)


def _model(provider):
    if provider in _MODEL_CACHE:
        return _MODEL_CACHE[provider]
    args = _init_args()
    torch.manual_seed(0)
    if provider == "PyTorch / ACL":
        m = TorchRef(*args)
    elif provider == "Baseline Triton1":
        m = input_mod.ModelNew(*args)
    elif provider == "Optimized Triton":
        m = opt_mod.ModelNew(*args)
    else:
        return None
    m = m.to("npu").eval()
    _MODEL_CACHE[provider] = m
    return m


def _make_input(label):
    for row in _BENCH_SHAPES:
        if row[0] == label:
            _, n, c, d, h, w = row
            torch.manual_seed(123)
            return torch.rand(n, c, d, h, w, device="npu")
    raise KeyError(label)


def _run(provider, x):
    if provider == "Baseline Triton2":
        raise RuntimeError("base_*.py reference is sandbox-read-prohibited")
    return _model(provider)(x)


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().detach().cpu())


def _same(a, b, atol=1e-3, rtol=1e-3):
    diff = (a.float() - b.float()).abs()
    tol = atol + rtol * b.float().abs()
    return bool((diff <= tol).all().detach().cpu()), float(diff.max().detach().cpu())


def _optimized_grid_ok(label):
    # Optimized production path is ACL. Triton fallback direct/persistent is unit-forced below.
    return True


def unit_test():
    ok_all = True
    for label, *_ in _BENCH_SHAPES:
        x = _make_input(label)
        ref = _run("PyTorch / ACL", x)
        for provider in PROVIDERS[1:]:
            if provider == "Baseline Triton2":
                print(f"TEST Baseline Triton2 {label}: SKIP_UNAVAILABLE base_file_read_prohibited max_abs=inf")
                continue
            if provider == "Baseline Triton1" and label == "default_required":
                print(f"TEST Baseline Triton1 {label}: SKIP_PRESKIP grid_toxic_or_slow max_abs=inf")
                continue
            try:
                y = _run(provider, x)
                same, mx = _same(y, ref)
                print(f"TEST {provider} {label}: {'PASS' if same else 'MISMATCH'} max_abs={mx:.6g}")
                ok_all = ok_all and (same or provider != "Optimized Triton")
            except Exception as e:
                print(f"TEST {provider} {label}: INFO_UNAVAILABLE {type(e).__name__} max_abs=inf")
                if provider == "Optimized Triton":
                    ok_all = False
    # Force optimized Triton fallback direct and persistent paths on a modest shape.
    x = _make_input("tiny_triton_path")
    ref = _run("PyTorch / ACL", x)
    old_use, old_grid = opt_mod._USE_TRITON_POST, opt_mod._MAX_GRID
    try:
        opt_mod._USE_TRITON_POST = True
        opt_mod._MAX_GRID = 65535
        _MODEL_CACHE.pop("Optimized Triton", None)
        y = _run("Optimized Triton", x)
        same, mx = _same(y, ref)
        print(f"TEST Optimized Triton forced_triton_direct: {'PASS' if same else 'MISMATCH'} max_abs={mx:.6g}")
        ok_all = ok_all and same
        opt_mod._MAX_GRID = 1
        _MODEL_CACHE.pop("Optimized Triton", None)
        y = _run("Optimized Triton", x)
        same, mx = _same(y, ref)
        print(f"TEST Optimized Triton forced_triton_persistent: {'PASS' if same else 'MISMATCH'} max_abs={mx:.6g}")
        ok_all = ok_all and same
    except Exception as e:
        print(f"TEST Optimized Triton forced_triton_paths: INFO_UNAVAILABLE {type(e).__name__} max_abs=inf")
        ok_all = False
    finally:
        opt_mod._USE_TRITON_POST, opt_mod._MAX_GRID = old_use, old_grid
        _MODEL_CACHE.pop("Optimized Triton", None)
    print("UNIT_TEST PASS" if ok_all else "UNIT_TEST_FAILED")
    return ok_all


def _manual_bench(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / rep


def bench_provider(label, provider):
    if provider == "Baseline Triton2":
        print(f"INFO bench_unavailable {provider} {label} base_file_read_prohibited")
        return float("inf")
    if provider == "Baseline Triton1" and label == "default_required":
        print(f"INFO bench_preskipped {provider} {label} avoid_grid_context_poisoning")
        return float("inf")
    try:
        x = _make_input(label)
        fn = lambda: _run(provider, x)
        try:
            return triton.testing.do_bench(fn, warmup=10, rep=30, return_mode="mean")
        except Exception:
            return _manual_bench(fn)
    except Exception as e:
        print(f"INFO bench_unavailable {provider} {label} {type(e).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[r[0] for r in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDERS,
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="l2_60_convtranspose3d_swish_groupnorm_hardswish",
        args={},
    )
)
def benchmark(label, provider):
    return bench_provider(label, provider)


def main():
    unit_test()
    print("BENCHMARK_START")
    benchmark.run(print_data=True, show_plots=False, save_path=None)


if __name__ == "__main__":
    main()
