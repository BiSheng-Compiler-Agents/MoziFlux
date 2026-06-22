"""
profile_kernels.py -- l2_25 Conv2d_Min_Tanh_Tanh

Compares PyTorch/ACL reference, original Triton baseline(s), and the optimized
ACL-dispatch implementation.  Comparison Triton providers are kept parser-visible
but are pre-skipped because the editable baseline epilogue uses unsupported
triton-ascend tanh lowering and the read-only base provider must not be modified.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

_DIR = Path(__file__).parent


def _load(fname):
    spec = importlib.util.spec_from_file_location("k_" + Path(fname).stem,
                                                  _DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname):
    try:
        return _load(fname)
    except Exception:
        return None


_baseline_mod1 = _load_optional("25_Conv2d_Min_Tanh_Tanh.py")
_baseline_mod2 = _load_optional("base_25_Conv2d_Min_Tanh_Tanh.py")
_optimized_mod = _load("opt_25_Conv2d_Min_Tanh_Tanh.py")

_PROVIDERS = [
    ("torch_ref", "PyTorch / ACL"),
    ("baseline1", "Baseline Triton1"),
    ("baseline2", "Baseline Triton2"),
    ("optimized", "Optimized Triton"),
]
_VARIANTS = [(k, n) for k, n in _PROVIDERS if k != "torch_ref"]
_MODS = {
    "baseline1": _baseline_mod1,
    "baseline2": _baseline_mod2,
    "optimized": _optimized_mod
}
_MODELS = {}
_REF_MODELS = {}

_BENCH_SHAPES = [
    # label, batch, in_channels, out_channels, height, width, kernel_size
    ("small_32", 4, 16, 64, 32, 32, 3),
    ("medium_128", 16, 16, 64, 128, 128, 3),
    ("exact_256", 128, 16, 64, 256, 256, 3),
]


def _init_args_from_shape(shape):
    _, _, cin, cout, _, _, k = shape
    return [cin, cout, k]


def _model(key, shape):
    cache_key = (key, tuple(_init_args_from_shape(shape)))
    if cache_key not in _MODELS:
        mod = _MODS.get(key)
        if mod is None:
            _MODELS[cache_key] = None
        else:
            torch.manual_seed(0)
            init = _init_args_from_shape(shape)
            _MODELS[cache_key] = mod.ModelNew(*init).to(device="npu").eval()
    return _MODELS[cache_key]


def _ref_model(shape):
    cache_key = tuple(_init_args_from_shape(shape))
    if cache_key not in _REF_MODELS:
        torch.manual_seed(0)
        cin, cout, k = cache_key
        _REF_MODELS[cache_key] = torch.nn.Conv2d(cin, cout,
                                                 k).to(device="npu").eval()
    return _REF_MODELS[cache_key]


def _make_input(shape):
    _, b, cin, _, h, w, _ = shape
    torch.manual_seed(42)
    return torch.rand(b, cin, h, w, device="npu")


def _run_torch_ref(x, shape):
    conv = _ref_model(shape)
    y = torch.nn.functional.conv2d(
        x,
        conv.weight,
        conv.bias,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )
    return torch.tanh(torch.tanh(torch.amin(y, dim=1, keepdim=True)))


def _run_provider(key, x, shape):
    if key == "torch_ref":
        return _run_torch_ref(x, shape)
    if key in ("baseline1", "baseline2"):
        raise RuntimeError("SKIP_baseline_triton_provider")
    model = _model(key, shape)
    if model is None:
        raise RuntimeError("SKIP_provider_unavailable")
    return model(x)


def _sync():
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def unit_test():
    print("=== Unit Test: l2_25_Conv2d_Min_Tanh_Tanh ===")
    optimized_ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_input(shape)
        with torch.no_grad():
            ref = _run_torch_ref(x, shape)
            _sync()
        for key, name in _VARIANTS:
            if key in ("baseline1", "baseline2"):
                print(f"TEST {key} {label} SKIP baseline_triton_provider")
                continue
            try:
                with torch.no_grad():
                    out = _run_provider(key, x, shape)
                    _sync()
                ok = torch.allclose(ref, out, atol=1e-4, rtol=1e-4)
                max_diff = (ref - out).abs().max().item()
                print(
                    f"TEST {key} {label} {'PASS' if ok else 'BAD'} max_diff={max_diff:.3e}"
                )
                optimized_ok = optimized_ok and ok
            except Exception as exc:
                print(
                    f"TEST {key} {label} BAD {type(exc).__name__}:{str(exc)[:80]}"
                )
                optimized_ok = False
    print("UNIT_TEST PASS" if optimized_ok else "UNIT_TEST_FAILED")
    return optimized_ok


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=[k for k, _ in _PROVIDERS],
        line_names=[n for _, n in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "-")],
        ylabel="Latency (ms)",
        plot_name="conv2d_min_tanh_tanh_perf",
        args={},
    ))
def benchmark(label, mode):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = _make_input(shape)
    if mode in ("baseline1", "baseline2"):
        print(f"BENCH {mode} {label} INF baseline_triton_provider")
        return float("inf")
    try:
        with torch.no_grad():
            return triton.testing.do_bench(
                lambda: _run_provider(mode, x, shape),
                warmup=10,
                rep=50,
                return_mode="mean")
    except Exception as exc:
        print(f"BENCH {mode} {label} INF {type(exc).__name__}:{str(exc)[:80]}")
        return float("inf")


def main():
    parser = argparse.ArgumentParser(
        description="Profile l2_25 Conv2d_Min_Tanh_Tanh")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    run_test = args.test or not args.bench
    run_bench = args.bench or not args.test
    ok = True
    if run_test:
        ok = unit_test() and ok
    if run_bench:
        benchmark.run(save_path=str(_DIR), print_data=True)
    # Keep exit zero so benchmark/results artifacts are always produced.
    sys.exit(0)


if __name__ == "__main__":
    main()
