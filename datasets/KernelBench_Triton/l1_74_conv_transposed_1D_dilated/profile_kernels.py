import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import triton

ROOT = Path(__file__).resolve().parent
_MODEL_CACHE = {}
_BENCH_SHAPES = [
    ("small_L256", 2, 32, 256),
    ("medium_L4096", 8, 32, 4096),
    ("irregular_L8193", 4, 32, 8193),
    ("exact_L131072", 32, 32, 131072),
]


def _load(stem):
    path = ROOT / stem
    name = "k_" + stem.replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_MODS = {
    "baseline1": None,
    "baseline2": None,
    "optimized": None,
}


def _module(key):
    if _MODS[key] is not None:
        return _MODS[key]
    if key == "baseline1":
        _MODS[key] = _load("74_conv_transposed_1D_dilated.py")
    elif key == "baseline2":
        p = ROOT / "base_74_conv_transposed_1D_dilated.py"
        _MODS[key] = _load(p.name) if p.exists() else False
    elif key == "optimized":
        _MODS[key] = _load("opt_74_conv_transposed_1D_dilated.py")
    return _MODS[key]


def _init_args():
    return [32, 64, 5, 1, 0, 3]


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    mod = _module(key)
    if mod is False:
        return None
    torch.manual_seed(0)
    m = mod.ModelNew(*_init_args()).to(device="npu").eval()
    _MODEL_CACHE[key] = m
    return m


def _torch_ref_model():
    torch.manual_seed(0)
    return nn.ConvTranspose1d(32,
                              64,
                              5,
                              stride=1,
                              padding=0,
                              dilation=3,
                              bias=False).to(device="npu").eval()


def _make_inputs(label, B, C, L):
    torch.manual_seed(123)
    x = torch.rand((B, C, L), device="npu", dtype=torch.float32)
    return [x]


def _run_torch_ref(x):
    return _torch_ref_model()(x)


def _run_provider(key, x):
    if key in ("baseline1", "baseline2"):
        raise RuntimeError("provider_preskip_custom_triton_convtranspose1d")
    return _model(key)(x)


def _sync():
    torch.npu.synchronize()


def _time_ms(fn, warmup=5, rep=20):
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


def unit_test():
    ok = True
    for label, B, C, L in _BENCH_SHAPES:
        x, = _make_inputs(label, B, C, L)
        with torch.no_grad():
            ref = _run_torch_ref(x)
            for key in ("baseline1", "baseline2"):
                print(f"TEST {key} {label} SKIP provider_preskip")
            out = _run_provider("optimized", x)
            max_diff = (out - ref).abs().max().item()
            if torch.allclose(out, ref, rtol=1e-3, atol=1e-3):
                print(f"TEST optimized {label} PASS max_diff={max_diff:.6g}")
            else:
                print(f"TEST optimized {label} FAIL max_diff={max_diff:.6g}")
                ok = False
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
        styles=[("blue", "-"), ("red", "--"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="conv_transpose1d_dilated",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    x, = _make_inputs(*shape)
    with torch.no_grad():
        if provider == "torch":
            return _time_ms(lambda: _run_torch_ref(x))
        if provider in ("baseline1", "baseline2"):
            return float("inf")
        return _time_ms(lambda: _run_provider("optimized", x))


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
