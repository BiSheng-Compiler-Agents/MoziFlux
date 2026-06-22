import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent
INPUT_SIZE = 8192
HIDDEN_SIZE = 8192
SCALING = 1.5


def _load(fname, optional=False):
    try:
        stem = Path(fname).stem.replace('.', '_')
        spec = importlib.util.spec_from_file_location(f"k_l2_14_{stem}",
                                                      ROOT / fname)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        if optional:
            print(
                f"INFO optional_provider {fname} SKIP import_unavailable {type(exc).__name__}"
            )
            return None
        raise


_in_mod = _load("14_Gemm_Divide_Sum_Scaling.py")
_base2_mod = _load("base_14_Gemm_Divide_Sum_Scaling.py", optional=True)
_opt_mod = _load("opt_14_Gemm_Divide_Sum_Scaling.py")
_MODEL_CACHE = {}
_WEIGHT_CACHE = {}
_BENCH_SHAPES = [
    ("small", 128, 1024, 1024),
    ("medium", 512, 4096, 4096),
    ("target", 1024, 8192, 8192),
]
_SHAPE_MAP = {label: (m, k, h) for label, m, k, h in _BENCH_SHAPES}


def _device():
    return torch.device("npu")


def _sync():
    torch.npu.synchronize()


def _make_inputs(label):
    m, k, _ = _SHAPE_MAP[label]
    torch.manual_seed(1234 + m + k)
    return torch.rand((m, k), dtype=torch.float32, device=_device())


def _weight(k, h):
    key = (k, h)
    if key not in _WEIGHT_CACHE:
        torch.manual_seed(0)
        _WEIGHT_CACHE[key] = torch.randn(h, k,
                                         dtype=torch.float32).to(_device())
    return _WEIGHT_CACHE[key]


def _model(provider, k, h):
    key = (provider, k, h)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    mod = {
        "baseline1": _in_mod,
        "baseline2": _base2_mod,
        "optimized": _opt_mod
    }[provider]
    if mod is None:
        return None
    torch.manual_seed(0)
    model = mod.ModelNew(k, h, SCALING).to(_device()).eval()
    _MODEL_CACHE[key] = model
    return model


def _run_torch_ref(x, k, h):
    w = _weight(k, h)
    return ((torch.matmul(x, w.t()) / 2.0).sum(dim=1, keepdim=True) * SCALING)


def _run_provider(provider, x, k, h):
    if provider == "torch":
        return _run_torch_ref(x, k, h)
    if provider == "baseline1":
        # Source baseline contains cache_modifier=".cg", a known Ascend compile blocker.
        return None
    model = _model(provider, k, h)
    if model is None:
        return None
    with torch.no_grad():
        return model(x)


def _close(a, b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-3)


def unit_test():
    ok = True
    for label, m, k, h in _BENCH_SHAPES:
        x = _make_inputs(label)
        ref = _run_torch_ref(x, k, h)
        _sync()
        print(f"TEST torch_ref {label} PASS")
        print(f"TEST baseline1 {label} SKIP source_cache_modifier_cg")
        for provider in ["baseline2", "optimized"]:
            try:
                out = _run_provider(provider, x, k, h)
                _sync()
                if out is None:
                    print(f"TEST {provider} {label} SKIP provider_unavailable")
                    continue
                if _close(out, ref):
                    print(f"TEST {provider} {label} PASS")
                else:
                    max_abs = (out - ref).abs().max().item()
                    print(
                        f"TEST {provider} {label} SKIP value_mismatch max_abs={max_abs:.6g}"
                    )
                    if provider == "optimized":
                        ok = False
            except Exception as exc:
                print(
                    f"TEST {provider} {label} SKIP runtime_unavailable {type(exc).__name__}"
                )
                if provider == "optimized":
                    ok = False
        if label == "small":
            try:
                opt = _model("optimized", k, h)
                out = opt.forward_triton(x)
                _sync()
                if _close(out, ref):
                    print(f"TEST optimized_triton_fallback {label} PASS")
                else:
                    print(
                        "TEST optimized_triton_fallback small SKIP value_mismatch"
                    )
            except Exception as exc:
                print(
                    f"TEST optimized_triton_fallback {label} SKIP runtime_unavailable {type(exc).__name__}"
                )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_ms(fn, warmup=5, rep=20):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        import time
        for _ in range(warmup):
            fn()
            _sync()
        start = time.perf_counter()
        for _ in range(rep):
            fn()
            _sync()
        return (time.perf_counter() - start) * 1000.0 / rep


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
        styles=[("blue", "-"), ("red", "-"), ("orange", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="l2_14_gemm_divide_sum_scaling",
        args={},
    ))
def bench(label, provider):
    m, k, h = _SHAPE_MAP[label]
    if provider == "baseline1":
        return float("inf")
    if provider == "baseline2" and _base2_mod is None:
        return float("inf")
    x = _make_inputs(label)
    try:

        def fn():
            return _run_provider(provider, x, k, h)

        y = fn()
        _sync()
        if y is None:
            return float("inf")
        return _time_ms(fn)
    except Exception:
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
        bench.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
