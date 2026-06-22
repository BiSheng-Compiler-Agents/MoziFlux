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
INPUT_FILE = "27_Conv3d_HardSwish_ReLU_Softmax_Mean.py"
BASE_FILE = "base_27_Conv3d_HardSwish_ReLU_Softmax_Mean.py"
OPT_FILE = "opt_27_Conv3d_HardSwish_ReLU_Softmax_Mean.py"

_BENCH_SHAPES = [
    ("small", 8, 3, 8, 16, 16, 16),
    ("medium", 32, 3, 12, 24, 24, 16),
    ("target", 128, 3, 16, 32, 32, 16),
]
_DISPATCH_ONLY_SHAPES = [
    ("tiny_triton", 2, 3, 3, 3, 3, 16),
    ("fallback_c80", 2, 3, 6, 8, 8, 80),
]
_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_PROVIDER_NAMES = [
    "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
]
_MODEL_CACHE = {}


def _load(fname: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname: str, name: str):
    try:
        return _load(fname, name)
    except Exception as exc:
        print(
            f"INFO optional_provider {fname} unavailable {type(exc).__name__}")
        return None


baseline1_mod = _load(INPUT_FILE, "k_baseline1_27")
baseline2_mod = _load_optional(BASE_FILE, "k_baseline2_27")
optimized_mod = _load(OPT_FILE, "k_optimized_27")


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _device():
    return torch.device("npu" if hasattr(torch, "npu")
                        and torch.npu.is_available() else "cuda")


def _make_input(N, in_ch, D, H, W, dtype=torch.float16):
    return torch.randn((N, in_ch, D, H, W), device=_device(), dtype=dtype)


def _init_args(in_ch, out_ch):
    return [in_ch, out_ch, 3]


def _make_torch_model(in_ch, out_ch, dtype):
    torch.manual_seed(0)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(0)
    return nn.Conv3d(in_ch, out_ch, 3).eval().to(device=_device(), dtype=dtype)


def _model(provider, in_ch, out_ch, dtype=torch.float16):
    key = (provider, in_ch, out_ch, str(dtype))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if provider == "torch":
        model = _make_torch_model(in_ch, out_ch, dtype)
    else:
        mod = {
            "baseline1": baseline1_mod,
            "baseline2": baseline2_mod,
            "optimized": optimized_mod
        }[provider]
        if mod is None:
            return None
        torch.manual_seed(0)
        if hasattr(torch, "npu"):
            torch.npu.manual_seed_all(0)
        model = mod.ModelNew(*_init_args(in_ch, out_ch)).eval().to(
            device=_device(), dtype=dtype)
    _MODEL_CACHE[key] = model
    return model


def _torch_ref(x, in_ch, out_ch):
    conv = _model("torch", in_ch, out_ch, x.dtype)
    with torch.no_grad():
        z = conv(x)
        y = torch.relu(z) * torch.clamp(z + 3.0, 0.0, 6.0) / 6.0
        return F.softmax(y, dim=1).mean(dim=(2, 3, 4))


def _run_provider(provider, x, in_ch, out_ch):
    if provider == "torch":
        return _torch_ref(x, in_ch, out_ch)
    model = _model(provider, in_ch, out_ch, x.dtype)
    if model is None:
        raise RuntimeError("provider_unavailable")
    with torch.no_grad():
        return model(x)


def _assert_close(label, provider, out, ref):
    out_f = out.float().detach().cpu()
    ref_f = ref.float().detach().cpu()
    torch.testing.assert_close(out_f, ref_f, rtol=1e-3, atol=1e-3)


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES + _DISPATCH_ONLY_SHAPES:
        label, N, in_ch, D, H, W, out_ch = shape
        x = _make_input(N, in_ch, D, H, W)
        ref = _torch_ref(x, in_ch, out_ch)
        for provider in ["baseline1", "baseline2", "optimized"]:
            try:
                out = _run_provider(provider, x, in_ch, out_ch)
                _sync()
                _assert_close(label, provider, out, ref)
                print(f"TEST {provider} {label} PASS")
            except Exception as exc:
                if provider == "optimized":
                    ok = False
                    print(
                        f"TEST optimized {label} MISMATCH {type(exc).__name__}: {exc}"
                    )
                else:
                    print(
                        f"TEST {provider} {label} SKIP value_diff_or_unavailable {type(exc).__name__}"
                    )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _manual_bench(fn, warmup=10, rep=50):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
            _sync()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
            _sync()
        t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / rep


def bench_one(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, N, in_ch, D, H, W, out_ch = shape
    if provider == "baseline2" and baseline2_mod is None:
        print(f"INFO bench {provider} {label} inf provider_unavailable")
        return float("inf")
    x = _make_input(N, in_ch, D, H, W)
    try:
        # optimized fallback path is covered in unit tests; benchmark table keeps target regimes only.
        def fn():
            return _run_provider(provider, x, in_ch, out_ch)

        if hasattr(triton, "testing") and hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn,
                                           warmup=10,
                                           rep=50,
                                           return_mode="mean")
        return _manual_bench(fn)
    except Exception as exc:
        print(f"INFO bench {provider} {label} inf {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=_PROVIDER_NAMES,
        styles=[("black", "-"), ("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="Latency (ms)",
        plot_name="conv3d_hardswish_relu_softmax_mean",
        args={},
    ))
def benchmark(label, provider):
    return bench_one(label, provider)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    run_test = args.test or not args.bench
    run_bench = args.bench or not args.test
    if run_test:
        unit_test()
    if run_bench:
        benchmark.run(print_data=True, show_plots=False, save_path=None)


if __name__ == "__main__":
    main()
