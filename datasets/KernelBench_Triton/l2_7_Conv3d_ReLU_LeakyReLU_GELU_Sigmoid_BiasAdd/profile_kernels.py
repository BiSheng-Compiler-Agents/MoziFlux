import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd.py"
OPT_FILE = ROOT / "opt_7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd.py"
# Sandbox says reference base_*.py files must not be read; keep parser-visible column only.
BASE2_FILE = ROOT / "base_7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd.py"

_BENCH_SHAPES = [
    ("small", 2, 4, 8, 8, 10, 10, 3),
    ("medium", 8, 8, 16, 16, 20, 20, 3),
    ("default", 64, 8, 32, 32, 64, 64, 3),
]
_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_DISPLAY = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}
_MODELS = {}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_BASELINE = None
_OPT = None


def _baseline_mod():
    global _BASELINE
    if _BASELINE is None:
        _BASELINE = _load(INPUT_FILE, "k_l2_7_baseline1")
    return _BASELINE


def _opt_mod():
    global _OPT
    if _OPT is None:
        _OPT = _load(OPT_FILE, "k_l2_7_optimized")
    return _OPT


class TorchRef(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        y = self.conv(x)
        y = F.relu(y)
        y = F.leaky_relu(y, negative_slope=0.01)
        y = F.gelu(y, approximate="none")
        y = torch.sigmoid(y)
        return y + self.bias


def _seed(seed=0):
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def _shape_init(shape):
    label, batch, in_c, out_c, depth, height, width, k = shape
    return [in_c, out_c, k, (out_c, 1, 1, 1)]


def _model(provider, shape):
    label = shape[0]
    key = (provider, label)
    if key in _MODELS:
        return _MODELS[key]
    init = _shape_init(shape)
    _seed(0)
    if provider == "torch":
        model = TorchRef(*init).eval()
    elif provider == "baseline1":
        model = _baseline_mod().ModelNew(*init).eval()
    elif provider == "optimized":
        model = _opt_mod().ModelNew(*init).eval()
    else:
        return None
    model = model.to(device="npu", dtype=torch.float32)
    _MODELS[key] = model
    return model


def _make_input(shape):
    label, batch, in_c, out_c, depth, height, width, k = shape
    _seed(123)
    return torch.rand(batch,
                      in_c,
                      depth,
                      height,
                      width,
                      device="npu",
                      dtype=torch.float32)


def _reference(x, shape):
    with torch.no_grad():
        return _model("torch", shape)(x)


def _skip_reason(provider, shape):
    label, batch, in_c, out_c, depth, height, width, k = shape
    out_d, out_h, out_w = depth - k + 1, height - k + 1, width - k + 1
    n_elements = batch * out_c * out_d * out_h * out_w
    if provider == "baseline2":
        return "sandbox_forbids_base_read"
    if provider == "baseline1" and triton.cdiv(n_elements, 1024) > 65535:
        return "comparison_provider_preskipped_grid_cap"
    return None


def _run_provider(provider, x, shape):
    reason = _skip_reason(provider, shape)
    if reason:
        raise RuntimeError(reason)
    with torch.no_grad():
        return _model(provider, shape)(x)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_input(shape)
        try:
            ref = _reference(x, shape)
        except Exception as e:
            print(
                f"TEST PyTorch / ACL {label}: FAIL {type(e).__name__} max_abs=inf"
            )
            ok = False
            continue
        print(f"TEST PyTorch / ACL {label}: PASS max_abs=0.000000e+00")
        for provider in ["baseline1", "baseline2", "optimized"]:
            display = _DISPLAY[provider]
            reason = _skip_reason(provider, shape)
            if reason and provider != "optimized":
                print(f"TEST {display} {label}: SKIP_{reason} max_abs=inf")
                continue
            try:
                out = _run_provider(provider, x, shape)
                diff = _max_abs(out, ref)
                passed = diff <= 1e-3
                print(
                    f"TEST {display} {label}: {'PASS' if passed else 'FAIL'} max_abs={diff:.6e}"
                )
                if provider == "optimized" and not passed:
                    ok = False
            except Exception as e:
                safe = type(e).__name__
                print(
                    f"TEST {display} {label}: {'FAIL' if provider == 'optimized' else 'SKIP_UNAVAILABLE'} {safe} max_abs=inf"
                )
                if provider == "optimized":
                    ok = False
    # Force the optimized persistent path on a modest tensor without allocating the huge default shape.
    try:
        opt = _opt_mod()
        old = opt._MAX_GRID
        opt._MAX_GRID = 1
        _MODELS.pop(("optimized", "small"), None)
        shape = _BENCH_SHAPES[0]
        x = _make_input(shape)
        ref = _reference(x, shape)
        out = _run_provider("optimized", x, shape)
        diff = _max_abs(out, ref)
        passed = diff <= 1e-3
        print(
            f"TEST Optimized Triton forced_persistent: {'PASS' if passed else 'FAIL'} max_abs={diff:.6e}"
        )
        ok = ok and passed
        opt._MAX_GRID = old
        _MODELS.pop(("optimized", "small"), None)
    except Exception as e:
        print(
            f"TEST Optimized Triton forced_persistent: FAIL {type(e).__name__} max_abs=inf"
        )
        ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_ms(fn, warmup=5, rep=20):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(max(1, min(warmup, 2))):
            fn()
            torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(max(1, min(rep, 5))):
            fn()
            torch.npu.synchronize()
        return (time.perf_counter() - start) * 1000.0 / max(1, min(rep, 5))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[_DISPLAY[p] for p in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="l2_7_conv3d_postops_biasadd",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    reason = _skip_reason(provider, shape)
    if reason:
        print(f"INFO bench_skip {_DISPLAY[provider]} {label}: {reason}")
        return float("inf")
    try:
        x = _make_input(shape)
        if provider == "torch":

            def fn():
                return _reference(x, shape)
        else:

            def fn():
                return _run_provider(provider, x, shape)

        return _time_ms(fn)
    except Exception as e:
        print(
            f"INFO bench_unavailable {_DISPLAY[provider]} {label}: {type(e).__name__}"
        )
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
        bench.run(print_data=True, show_plots=False, save_path=str(ROOT))


if __name__ == "__main__":
    main()
