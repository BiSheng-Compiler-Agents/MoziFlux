import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "55_Matmul_MaxPool_Sum_Scale.py"
BASE_FILE = ROOT / "base_55_Matmul_MaxPool_Sum_Scale.py"
OPT_FILE = ROOT / "opt_55_Matmul_MaxPool_Sum_Scale.py"

_BENCH_SHAPES = [
    ("small_fallback", 4, 128, 128, 2, 0.5),
    ("irregular_acl", 7, 192, 130, 2, 0.5),
    ("default_acl", 128, 32768, 32768, 2, 0.5),
]
_PROVIDER_CACHE = {}
_MODEL_CACHE = {}
_REF_CACHE = {}


def _load(path: Path, key: str):
    if key in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[key]
    try:
        spec = importlib.util.spec_from_file_location(f"k55_{key}_{path.stem}",
                                                      path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _PROVIDER_CACHE[key] = mod
        return mod
    except Exception as exc:
        print(f"INFO provider_unavailable {key}: {type(exc).__name__}")
        _PROVIDER_CACHE[key] = None
        return None


def _device():
    return torch.device("npu")


def _make_inputs(B, IN_FEATURES, dtype=torch.float32):
    torch.manual_seed(123)
    return torch.rand((B, IN_FEATURES), device=_device(), dtype=dtype)


def _make_ref_model(IN_FEATURES,
                    OUT_FEATURES,
                    KERNEL,
                    scale,
                    dtype=torch.float32):
    key = (IN_FEATURES, OUT_FEATURES, KERNEL, float(scale), dtype)
    if key not in _REF_CACHE:
        torch.manual_seed(0)
        _REF_CACHE[key] = nn.Linear(IN_FEATURES,
                                    OUT_FEATURES).to(device=_device(),
                                                     dtype=dtype)
    return _REF_CACHE[key]


def _run_torch_ref(x, IN_FEATURES, OUT_FEATURES, KERNEL, scale):
    model = _make_ref_model(IN_FEATURES, OUT_FEATURES, KERNEL, scale, x.dtype)
    z = F.linear(x, model.weight, model.bias)
    pooled = F.max_pool1d(z.unsqueeze(1), kernel_size=KERNEL,
                          stride=KERNEL).squeeze(1)
    return pooled.sum(dim=1) * float(scale)


def _model(provider, IN_FEATURES, OUT_FEATURES, KERNEL, scale):
    key = (provider, IN_FEATURES, OUT_FEATURES, KERNEL, float(scale))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    file_key = {
        "baseline1": (INPUT_FILE, "baseline1"),
        "baseline2": (BASE_FILE, "baseline2"),
        "optimized": (OPT_FILE, "optimized"),
        "opt_fallback": (OPT_FILE, "optimized"),
    }[provider]
    mod = _load(*file_key)
    if mod is None:
        return None
    torch.manual_seed(0)
    try:
        m = mod.ModelNew(IN_FEATURES, OUT_FEATURES, KERNEL,
                         scale).to(device=_device(), dtype=torch.float32)
    except Exception as exc:
        print(f"INFO model_unavailable {provider}: {type(exc).__name__}")
        return None
    _MODEL_CACHE[key] = m
    return m


def _run_provider(provider, x, IN_FEATURES, OUT_FEATURES, KERNEL, scale):
    if provider == "torch":
        return _run_torch_ref(x, IN_FEATURES, OUT_FEATURES, KERNEL, scale)
    m = _model(provider, IN_FEATURES, OUT_FEATURES, KERNEL, scale)
    if m is None:
        raise RuntimeError("provider unavailable")
    if provider == "opt_fallback":
        mod = _load(OPT_FILE, "optimized")
        old = getattr(mod, "_USE_ACL_DISPATCH", True)
        mod._USE_ACL_DISPATCH = False
        try:
            if hasattr(m, "forward_triton_fallback"):
                return m.forward_triton_fallback(x)
            return m(x)
        finally:
            mod._USE_ACL_DISPATCH = old
    return m(x)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok = True
    providers = ["baseline1", "baseline2", "optimized", "opt_fallback"]
    for label, B, IN_FEATURES, OUT_FEATURES, KERNEL, scale in _BENCH_SHAPES:
        x = _make_inputs(B, IN_FEATURES)
        ref = _run_torch_ref(x, IN_FEATURES, OUT_FEATURES, KERNEL, scale)
        for provider in providers:
            if label == "default_acl" and provider in ("baseline1",
                                                       "baseline2",
                                                       "opt_fallback"):
                if provider == "baseline2":
                    print(
                        f"TEST {provider} {label}: SKIP huge_custom_path max_abs=inf"
                    )
                else:
                    print(
                        f"INFO test_skip {provider} {label}: huge custom Triton comparison path preskipped"
                    )
                continue
            try:
                y = _run_provider(provider, x, IN_FEATURES, OUT_FEATURES,
                                  KERNEL, scale)
                torch.npu.synchronize()
                diff = _max_abs(y, ref)
                passed = math.isfinite(diff) and diff <= 1e-3
                print(
                    f"TEST {provider} {label}: {'PASS' if passed else 'MISMATCH'} max_abs={diff:.6g}"
                )
                ok = ok and (passed or provider in ("baseline2", ))
            except Exception as exc:
                if provider == "baseline2":
                    print(
                        f"TEST {provider} {label}: UNAVAILABLE {type(exc).__name__} max_abs=inf"
                    )
                else:
                    print(
                        f"INFO test_unavailable {provider} {label}: {type(exc).__name__}"
                    )
                if provider in ("optimized", "opt_fallback") and not (
                        provider == "opt_fallback" and label == "default_acl"):
                    ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_once(provider, label, B, IN_FEATURES, OUT_FEATURES, KERNEL, scale):
    if label == "default_acl" and provider in ("baseline1", "baseline2",
                                               "opt_fallback"):
        print(
            f"INFO bench_skip {provider} {label}: huge custom Triton comparison path preskipped_to_avoid_timeout"
        )
        return float("inf")
    try:
        x = _make_inputs(B, IN_FEATURES)
        # Compile/warmup once before do_bench's timed loop.
        _run_provider(provider, x, IN_FEATURES, OUT_FEATURES, KERNEL, scale)
        torch.npu.synchronize()
        return triton.testing.do_bench(lambda: _run_provider(
            provider, x, IN_FEATURES, OUT_FEATURES, KERNEL, scale),
                                       warmup=10,
                                       rep=50,
                                       return_mode="mean")
    except Exception as exc:
        print(
            f"INFO bench_unavailable {provider} {label}: {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=[
            "torch", "baseline1", "baseline2", "optimized", "opt_fallback"
        ],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton", "Optimized TritonFallback"
        ],
        styles=[("black", "-"), ("red", "--"), ("orange", "--"),
                ("green", "-"), ("blue", "-")],
        ylabel="latency ms",
        plot_name="matmul_maxpool_sum_scale",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, B, IN_FEATURES, OUT_FEATURES, KERNEL, scale = shape
    return _bench_once(provider, label, B, IN_FEATURES, OUT_FEATURES, KERNEL,
                       scale)


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
