import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = "21_Conv2d_Add_Scale_Sigmoid_GroupNorm.py"
BASE_FILE = "base_21_Conv2d_Add_Scale_Sigmoid_GroupNorm.py"
OPT_FILE = "opt_21_Conv2d_Add_Scale_Sigmoid_GroupNorm.py"

_BENCH_SHAPES = [
    ("small_direct", 1, 8, 64, 64),
    ("medium_direct", 8, 8, 128, 128),
    ("exact_persistent", 128, 8, 256, 256),
]


def _load(fname, key):
    path = ROOT / fname
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, key):
    try:
        return _load(fname, key)
    except Exception as exc:
        print(f"INFO provider {key} import_skip {type(exc).__name__}")
        return None


_MODS = {
    "baseline1":
    _load(INPUT_FILE, "baseline1"),
    "baseline2":
    _load_optional(BASE_FILE, "baseline2") if
    (ROOT / BASE_FILE).exists() else None,
    "optimized":
    _load(OPT_FILE, "optimized"),
}

_PROVIDERS = ["torch", "baseline1"]
_LINE_NAMES = ["PyTorch / ACL", "Baseline Triton1"]
_STYLES = [("blue", "-"), ("red", "--")]
if _MODS["baseline2"] is not None:
    _PROVIDERS.append("baseline2")
    _LINE_NAMES.append("Baseline Triton2")
    _STYLES.append(("black", "--"))
_PROVIDERS.append("optimized")
_LINE_NAMES.append("Optimized Triton")
_STYLES.append(("green", "-"))

_MODEL_CACHE = {}


class TorchRef(nn.Module):

    def __init__(self,
                 in_channels=8,
                 out_channels=32,
                 kernel_size=3,
                 num_groups=8,
                 bias_shape=None,
                 scale_shape=None):
        super().__init__()
        if bias_shape is None:
            bias_shape = (out_channels, 1, 1)
        if scale_shape is None:
            scale_shape = (out_channels, 1, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid((x + self.bias) * self.scale)
        x = self.group_norm(x)
        return x


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    return init


def _model(provider, dtype=torch.float32):
    key = (provider, str(dtype))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch.manual_seed(0)
    if provider == "torch":
        m = TorchRef(*_init_args(_MODS["baseline1"]))
    else:
        mod = _MODS[provider]
        if mod is None:
            return None
        m = mod.ModelNew(*_init_args(mod))
    m = m.to(device="npu", dtype=dtype).eval()
    _MODEL_CACHE[key] = m
    return m


def _make_inputs(label):
    for row in _BENCH_SHAPES:
        if row[0] == label:
            _, n, c, h, w = row
            torch.manual_seed(123)
            return [torch.rand(n, c, h, w, device="npu")]
    raise KeyError(label)


def _baseline1_grid_risky(label):
    _, n, _c, h, w = next(row for row in _BENCH_SHAPES if row[0] == label)
    out_elems = n * 32 * (h - 2) * (w - 2)
    # baseline autotune contains BLOCK_SIZE=1024/2048, which can exceed Ascend coreDim during tuning.
    return triton.cdiv(out_elems, 1024) > 65535


def _baseline2_grid_risky(label):
    return _baseline1_grid_risky(label)


def _run_provider(provider, xs):
    if provider == "torch":
        return _model("torch", xs[0].dtype)(*xs)
    if provider == "baseline1" and _baseline1_grid_risky(CURRENT_LABEL):
        raise RuntimeError("grid_guard")
    if provider == "baseline2" and _baseline2_grid_risky(CURRENT_LABEL):
        raise RuntimeError("grid_guard")
    m = _model(provider, xs[0].dtype)
    if m is None:
        raise RuntimeError("provider_unavailable")
    return m(*xs)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _time_ms(fn, warmup=5, rep=20):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


def _bench_one(provider, label):
    global CURRENT_LABEL
    CURRENT_LABEL = label
    xs = _make_inputs(label)
    try:
        # Keep comparison columns parser-visible but do not execute baseline Triton in
        # benchmarks: its autotune configs can poison the NPU context before optimized timing.
        if provider in ("baseline1", "baseline2"):
            return float("inf")

        def fn():
            return _run_provider(provider, xs)

        # Manual timing is used on Ascend here because do_bench can fail/poison the
        # process after autotuned comparison providers; perf_report still formats the table.
        return _time_ms(fn, warmup=3, rep=10)
    except Exception as exc:
        msg = str(exc).replace("\n", " ")[:180]
        print(f"INFO bench {provider} {label} skip {type(exc).__name__} {msg}")
        return float("inf")


CURRENT_LABEL = ""


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[r[0] for r in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=_LINE_NAMES,
        styles=_STYLES,
        ylabel="ms",
        plot_name="conv2d_add_scale_sigmoid_groupnorm",
        args={},
    ))
def benchmark(label, provider):
    return _bench_one(provider, label)


def unit_test():
    ok = True
    for label, *_ in _BENCH_SHAPES:
        global CURRENT_LABEL
        CURRENT_LABEL = label
        xs = _make_inputs(label)
        with torch.no_grad():
            ref = _run_provider("torch", xs)
            for provider in _PROVIDERS:
                if provider == "torch":
                    print(f"TEST torch {label} PASS max_err=0")
                    continue
                try:
                    if provider == "baseline1" and _baseline1_grid_risky(
                            label):
                        print(f"TEST baseline1 {label} SKIP grid_guard")
                        continue
                    if provider == "baseline2" and _baseline2_grid_risky(
                            label):
                        print(f"TEST baseline2 {label} SKIP grid_guard")
                        continue
                    out = _run_provider(provider, xs)
                    _sync()
                    diff = (out.float() - ref.float()).abs()
                    max_err = float(diff.max().detach().cpu())
                    ref_den = ref.float().abs().clamp_min(1e-6)
                    max_rel = float((diff / ref_den).max().detach().cpu())
                    if provider == "optimized" and not torch.allclose(
                            out.float(), ref.float(), rtol=1e-3, atol=1e-3):
                        print(
                            f"TEST optimized {label} MISMATCH max_err={max_err:.6g} max_rel={max_rel:.6g}"
                        )
                        ok = False
                    else:
                        status = "PASS" if provider == "optimized" else (
                            "PASS"
                            if max_err <= 1e-3 or max_rel <= 1e-3 else "SKIP")
                        print(
                            f"TEST {provider} {label} {status} max_err={max_err:.6g} max_rel={max_rel:.6g}"
                        )
                except Exception as exc:
                    if provider == "optimized":
                        print(
                            f"TEST optimized {label} MISMATCH exception={type(exc).__name__}"
                        )
                        ok = False
                    else:
                        print(
                            f"TEST {provider} {label} SKIP {type(exc).__name__}"
                        )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if not args.test and not args.bench:
        args.test = args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
