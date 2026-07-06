import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool.py"
OPT_FILE = HERE / "opt_77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool.py"
BASE2_FILE = HERE / "base_77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool.py"

_BENCH_SHAPES = [
    ("tiny_direct", 2, 8, 4, 8, 8, 4, 3),
    ("irregular_direct", 3, 7, 5, 6, 7, 5, 3),
    ("default_required", 16, 64, 16, 32, 32, 128, 5),
]
SHAPES = {s[0]: s for s in _BENCH_SHAPES}
PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
DISPLAY = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}
_MODS = {}
_MODELS = {}


class RefModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale_factor
        x = self.batch_norm(x)
        return self.global_avg_pool(x)


def _load(path: Path, key: str):
    if key in _MODS:
        return _MODS[key]
    spec = importlib.util.spec_from_file_location(f"k_l2_77_{key}", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODS[key] = mod
    return mod


def _shape(label):
    _, n, cin, d, h, w, cout, k = SHAPES[label]
    return n, cin, d, h, w, cout, k


def _make_input(label, device):
    n, cin, d, h, w, _, _ = _shape(label)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000 + len(label))
    return torch.rand((n, cin, d, h, w), generator=gen, dtype=torch.float32).to(device)


def _new_model(provider, label, device):
    n, cin, d, h, w, cout, k = _shape(label)
    scale = 2.0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        if provider == "torch":
            model = RefModel(cin, cout, k, scale)
        elif provider == "baseline1":
            model = _load(INPUT_FILE, "baseline1").ModelNew(cin, cout, k, scale)
        elif provider == "optimized":
            model = _load(OPT_FILE, "optimized").ModelNew(cin, cout, k, scale)
        else:
            return None
    return model.to(device=device, dtype=torch.float32).eval()


def _model(provider, label, device):
    key = (provider, label, str(device))
    if key not in _MODELS:
        _MODELS[key] = _new_model(provider, label, device)
    return _MODELS[key]


def _run_provider(provider, label, x):
    if provider == "baseline2":
        # Sandbox marks base_*.py as reference-only: keep parser-visible column without reading it.
        raise RuntimeError("BASELINE2_SANDBOXED")
    model = _model(provider, label, x.device)
    with torch.no_grad():
        return model(x)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().detach().cpu().item()


def unit_test():
    device = torch.device("npu:0")
    if hasattr(torch, "npu"):
        torch.npu.set_device(device)
    ok = True
    for label, *_ in _BENCH_SHAPES:
        x = _make_input(label, device)
        ref = _run_provider("torch", label, x)
        _sync()
        for provider in PROVIDERS[1:]:
            name = DISPLAY[provider]
            if provider == "baseline2":
                print(f"TEST {name} {label}: SKIP_UNAVAILABLE sandboxed_reference_file max_abs=inf")
                continue
            try:
                got = _run_provider(provider, label, x)
                _sync()
                diff = _max_abs(got, ref)
                passed = diff <= 5e-2 + 5e-3 * ref.float().abs().max().detach().cpu().item()
                print(f"TEST {name} {label}: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}")
                ok = ok and (passed or provider == "baseline1")
            except Exception as exc:
                print(f"TEST {name} {label}: SKIP_UNAVAILABLE {type(exc).__name__} max_abs=inf")
                ok = ok and (provider != "optimized")

    # Exercise optimized persistent and ACL fallback dispatches on a small shape.
    opt = _load(OPT_FILE, "optimized")
    x = _make_input("tiny_direct", device)
    ref = _run_provider("torch", "tiny_direct", x)
    old_grid = opt._MAX_GRID
    old_shortcut = opt._USE_SUM_SHORTCUT
    try:
        opt._MAX_GRID = 1
        _MODELS.pop(("optimized", "tiny_direct", str(device)), None)
        got = _run_provider("optimized", "tiny_direct", x)
        _sync()
        diff = _max_abs(got, ref)
        print(f"TEST Optimized Triton forced_persistent: {'PASS' if diff <= 5e-2 else 'FAIL'} max_abs={diff:.6g}")
        ok = ok and diff <= 5e-2
    finally:
        opt._MAX_GRID = old_grid
        _MODELS.pop(("optimized", "tiny_direct", str(device)), None)
    try:
        opt._USE_SUM_SHORTCUT = False
        _MODELS.pop(("optimized", "tiny_direct", str(device)), None)
        got = _run_provider("optimized", "tiny_direct", x)
        _sync()
        diff = _max_abs(got, ref)
        print(f"TEST Optimized Triton acl_fallback: {'PASS' if diff <= 5e-2 else 'FAIL'} max_abs={diff:.6g}")
        ok = ok and diff <= 5e-2
    finally:
        opt._USE_SUM_SHORTCUT = old_shortcut
        _MODELS.pop(("optimized", "tiny_direct", str(device)), None)
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_provider(label, provider):
    if provider == "baseline2":
        print(f"INFO benchmark {DISPLAY[provider]} {label}: inf sandboxed_reference_file")
        return float("inf")
    device = torch.device("npu:0")
    x = _make_input(label, device)
    try:
        # Build and correctness-check once before timing.
        ref = _run_provider("torch", label, x)
        got = _run_provider(provider, label, x)
        _sync()
        if provider != "torch":
            diff = _max_abs(got, ref)
            if not math.isfinite(diff) or diff > 1e-1 + 1e-2 * ref.float().abs().max().detach().cpu().item():
                print(f"INFO benchmark {DISPLAY[provider]} {label}: inf correctness_preskip max_abs={diff:.6g}")
                return float("inf")

        def fn():
            with torch.no_grad():
                _run_provider(provider, label, x)
            _sync()

        if hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn, warmup=1, rep=3, return_mode="mean")
        for _ in range(1):
            fn()
        t0 = time.perf_counter()
        for _ in range(3):
            fn()
        return (time.perf_counter() - t0) * 1000.0 / 3.0
    except Exception as exc:
        print(f"INFO benchmark {DISPLAY[provider]} {label}: inf {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=[DISPLAY[p] for p in PROVIDERS],
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="l2_77_convtranspose3d_scale_bn_gap",
        args={},
    )
)
def benchmark(label, provider):
    return _bench_provider(label, provider)


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
        benchmark.run(print_data=True, show_plots=False, save_path=str(HERE / "remote_results" / "profile_plots"))


if __name__ == "__main__":
    main()
