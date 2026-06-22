import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton

HERE = Path(__file__).resolve().parent
BASELINE1 = "82_conv_depthwise_2D_square_input_square_kernel.py"
BASELINE2 = "base_82_conv_depthwise_2D_square_input_square_kernel.py"
OPTIMIZED = "opt_82_conv_depthwise_2D_square_input_square_kernel.py"

_BENCH_SHAPES = [
    ("small_square", 1, 64, 32, 32),
    ("medium_square", 4, 64, 128, 128),
    ("target_square", 16, 64, 512, 512),
]

_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_MODEL_CACHE = {}
_REF_CACHE = {}
_LOAD_CACHE = {}
_SKIP = {}


def _load(fname, optional=False):
    if fname in _LOAD_CACHE:
        return _LOAD_CACHE[fname]
    path = HERE / fname
    try:
        spec = importlib.util.spec_from_file_location(
            "k_" + fname.replace(".", "_").replace("-", "_"), path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _LOAD_CACHE[fname] = mod
        return mod
    except Exception as exc:
        if optional:
            _SKIP["baseline2"] = f"import_unavailable:{type(exc).__name__}"
            _LOAD_CACHE[fname] = None
            return None
        raise


def _init_args():
    mod = _load(BASELINE1)
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    return list(init)


def _torch_ref(device):
    key = (device, tuple(_init_args()))
    if key not in _REF_CACHE:
        in_channels, kernel_size, stride, padding = _init_args()
        torch.manual_seed(0)
        ref = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        ).to(device=device).eval()
        _REF_CACHE[key] = ref
    return _REF_CACHE[key]


def _model(provider, device):
    if provider == "torch":
        return _torch_ref(device)
    if provider == "baseline1":
        fname = BASELINE1
    elif provider == "baseline2":
        fname = BASELINE2
    elif provider == "optimized":
        fname = OPTIMIZED
    else:
        raise KeyError(provider)
    key = (provider, device, tuple(_init_args()))
    if key not in _MODEL_CACHE:
        mod = _load(fname, optional=(provider == "baseline2"))
        if mod is None:
            return None
        torch.manual_seed(0)
        _MODEL_CACHE[key] = mod.ModelNew(*_init_args()).to(
            device=device).eval()
    return _MODEL_CACHE[key]


def _make_inputs(label, n, c, h, w, device):
    torch.manual_seed(123)
    return torch.rand((n, c, h, w), device=device, dtype=torch.float32)


def _grid_guard_baseline(n, c, h, w):
    # Baseline launches grid=(N*C, H_OUT, ceil(W_OUT/256)); Ascend launch product must be <= 65535.
    k, stride, pad = 3, 1, 0
    h_out = (h + 2 * pad - k) // stride + 1
    w_out = (w + 2 * pad - k) // stride + 1
    return n * c * h_out * triton.cdiv(w_out, 256)


def _run_provider(provider, x):
    if provider in ("baseline1",
                    "baseline2") and _grid_guard_baseline(*x.shape) > 65535:
        raise RuntimeError("grid_guard")
    model = _model(provider, x.device)
    if model is None:
        raise RuntimeError(_SKIP.get(provider, "provider_unavailable"))
    with torch.no_grad():
        return model(x)


def _allclose(a, b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-3)


def unit_test():
    device = "npu"
    ok = True
    for label, n, c, h, w in _BENCH_SHAPES:
        x = _make_inputs(label, n, c, h, w, device)
        ref = _run_provider("torch", x)
        for provider in ["baseline1", "baseline2", "optimized"]:
            try:
                y = _run_provider(provider, x)
                if _allclose(y, ref):
                    print(f"TEST {provider} {label} PASS")
                else:
                    max_err = (y - ref).abs().max().item()
                    if provider == "optimized":
                        print(
                            f"TEST {provider} {label} MISMATCH max_err={max_err:.6g}"
                        )
                        ok = False
                    else:
                        print(
                            f"TEST {provider} {label} SKIP value_delta max_err={max_err:.6g}"
                        )
            except Exception as exc:
                reason = str(exc).splitlines()[0].replace(" ", "_")[:80]
                if provider == "optimized":
                    print(f"TEST {provider} {label} ERROR {reason}")
                    ok = False
                else:
                    print(f"TEST {provider} {label} SKIP {reason}")
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _time_ms(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
        _sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
        _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="depthwise_conv2d_square",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, n, c, h, w = shape
    device = "npu"
    x = _make_inputs(label, n, c, h, w, device)
    if provider in ("baseline1", "baseline2") and _grid_guard_baseline(
            n, c, h, w) > 65535:
        return float("inf")
    try:
        return _time_ms(lambda: _run_provider(provider, x))
    except Exception:
        return float("inf")


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
        bench.run(print_data=True, show_plots=False, save_path=str(HERE))


if __name__ == "__main__":
    main()
