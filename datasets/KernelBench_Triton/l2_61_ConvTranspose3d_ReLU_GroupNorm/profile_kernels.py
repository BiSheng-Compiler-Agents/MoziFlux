import argparse
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
INPUT_FILE = ROOT / "61_ConvTranspose3d_ReLU_GroupNorm.py"
OPT_FILE = ROOT / "opt_61_ConvTranspose3d_ReLU_GroupNorm.py"
# base_*.py is intentionally not read in this sandbox; Baseline Triton2 remains parser-visible.

_BENCH_SHAPES = [
    ("tiny", 1, 64, 4, 8, 8),
    ("medium", 2, 64, 8, 16, 16),
    ("target", 16, 64, 32, 32, 32),
]

_LINE_VALS = ["torch", "baseline1", "baseline2", "optimized"]
_LINE_NAMES = [
    "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
]
_STYLES = [("blue", "-"), ("red", "--"), ("black", "--"), ("green", "-")]
_MODEL_CACHE = {}
_MOD_CACHE = {}


def _load(path: Path, key: str):
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _MOD_CACHE[key] = mod
    return mod


class TorchRef(nn.Module):

    def __init__(self,
                 in_channels=64,
                 out_channels=128,
                 kernel_size=3,
                 groups=8,
                 bias=False,
                 eps=1e-5):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups,
                                       num_channels=out_channels,
                                       eps=eps)

    def forward(self, x):
        y = self.conv_transpose(x)
        y = torch.relu(y)
        return F.group_norm(y, self.group_norm.num_groups,
                            self.group_norm.weight, self.group_norm.bias,
                            self.group_norm.eps)


def _make_inputs(label):
    rec = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, n, c, d, h, w = rec
    torch.manual_seed(123)
    return (torch.rand(n, c, d, h, w, device="npu", dtype=torch.float32), )


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    if key == "torch":
        model = TorchRef().to("npu").eval()
    elif key == "baseline1":
        mod = _load(INPUT_FILE, "baseline1")
        model = mod.ModelNew(*mod.get_init_inputs()).to("npu").eval()
    elif key == "optimized":
        mod = _load(OPT_FILE, "optimized")
        model = mod.ModelNew(*mod.get_init_inputs()).to("npu").eval()
    else:
        model = None
    _MODEL_CACHE[key] = model
    return model


def _run_torch_ref(x):
    with torch.no_grad():
        return _model("torch")(x)


def _run_provider(provider, x):
    if provider == "baseline2":
        raise RuntimeError(
            "Baseline Triton2 skipped: sandbox forbids reading base_*.py")
    with torch.no_grad():
        return _model(provider)(x)


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().detach().cpu())


def unit_test():
    ok = True
    for label, *_ in _BENCH_SHAPES:
        x, = _make_inputs(label)
        ref = _run_torch_ref(x)
        for provider, name in zip(_LINE_VALS, _LINE_NAMES):
            if provider == "torch":
                print(f"TEST {name} {label}: PASS max_abs=0.0")
                continue
            if provider == "baseline2":
                print(
                    f"TEST {name} {label}: SKIP_UNAVAILABLE sandbox_base_read_prohibited max_abs=inf"
                )
                continue
            if provider == "baseline1" and label == "target":
                print(
                    f"TEST {name} {label}: SKIP_COMPARISON target_preskipped_to_bound_runtime max_abs=inf"
                )
                continue
            try:
                y = _run_provider(provider, x)
                diff = _max_abs(y, ref)
                passed = math.isfinite(diff) and diff <= 1e-3
                print(
                    f"TEST {name} {label}: {'PASS' if passed else 'MISMATCH'} max_abs={diff:.6g}"
                )
                if provider == "optimized" and not passed:
                    ok = False
            except Exception as exc:
                tag = type(exc).__name__
                print(
                    f"TEST {name} {label}: SKIP_UNAVAILABLE {tag} max_abs=inf")
                if provider == "optimized":
                    ok = False
    ok = _test_optimized_triton_fallbacks() and ok
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _test_optimized_triton_fallbacks():
    mod = _load(OPT_FILE, "optimized")
    old_acl = getattr(mod, "_USE_ACL_DISPATCH", True)
    old_grid = getattr(mod, "_MAX_GRID", 65535)
    passed = True
    for forced_grid, path_name in [(65535, "direct"), (1, "persistent")]:
        try:
            setattr(mod, "_USE_ACL_DISPATCH", False)
            setattr(mod, "_MAX_GRID", forced_grid)
            _MODEL_CACHE.pop("optimized", None)
            x, = _make_inputs("tiny")
            y = _run_provider("optimized", x)
            ref = _run_torch_ref(x)
            diff = _max_abs(y, ref)
            path_ok = math.isfinite(diff) and diff <= 1e-3
            print(
                f"TEST Optimized Triton fallback_{path_name}: {'PASS' if path_ok else 'MISMATCH'} max_abs={diff:.6g}"
            )
            passed = passed and path_ok
        except Exception as exc:
            print(
                f"TEST Optimized Triton fallback_{path_name}: SKIP_UNAVAILABLE {type(exc).__name__} max_abs=inf"
            )
            passed = False
    setattr(mod, "_USE_ACL_DISPATCH", old_acl)
    setattr(mod, "_MAX_GRID", old_grid)
    _MODEL_CACHE.pop("optimized", None)
    return passed


def _time_ms(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / rep


def _bench_one(provider, label):
    if provider == "baseline2":
        print(
            f"INFO benchmark_preskip Baseline Triton2 {label}: sandbox_base_read_prohibited"
        )
        return float("inf")
    if provider == "baseline1" and label == "target":
        print(
            f"INFO benchmark_preskip Baseline Triton1 {label}: target_runtime_bounded"
        )
        return float("inf")
    x, = _make_inputs(label)
    try:

        def fn():
            return _run_torch_ref(x) if provider == "torch" else _run_provider(
                provider, x)

        reps = {"tiny": 20, "medium": 5, "target": 1}[label]
        warmup = {"tiny": 5, "medium": 2, "target": 0}[label]
        return _time_ms(fn, warmup=warmup, rep=reps)
    except Exception as exc:
        print(
            f"INFO benchmark_unavailable {_LINE_NAMES[_LINE_VALS.index(provider)]} {label}: {type(exc).__name__}"
        )
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_LINE_VALS,
        line_names=_LINE_NAMES,
        styles=_STYLES,
        ylabel="ms",
        plot_name="convtranspose3d_relu_groupnorm",
        args={},
    ))
def benchmark(label, provider):
    return _bench_one(provider, label)


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
