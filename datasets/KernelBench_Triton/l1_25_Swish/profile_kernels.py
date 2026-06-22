import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
_BENCH_SHAPES = [
    ("direct_small", (1024, 1024)),
    ("direct_medium", (4096, 8192)),
    ("persistent_original", (4096, 393216)),
]
_PROVIDER_FILES = {
    "Baseline Triton1": "25_Swish.py",
    "Baseline Triton2": "base_25_Swish.py",
    "Optimized Triton": "opt_25_Swish.py",
}
_MODEL_CACHE = {}
_MODULE_CACHE = {}


class TorchSwish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


def _load(key):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    path = ROOT / _PROVIDER_FILES[key]
    spec = importlib.util.spec_from_file_location(
        f"k_{key.replace(' ', '_').replace('/', '_')}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def _init_args(mod):
    if not hasattr(mod, "get_init_inputs"):
        return []
    init = mod.get_init_inputs()
    if init == [()]:
        return []
    return init or []


def _model(key):
    if key not in _MODEL_CACHE:
        if key == "PyTorch / ACL":
            model = TorchSwish()
        else:
            mod = _load(key)
            model = mod.ModelNew(*_init_args(mod))
        _MODEL_CACHE[key] = model.npu().eval()
    return _MODEL_CACHE[key]


def _make_inputs(shape):
    torch.manual_seed(0)
    x = torch.rand(shape, device="npu", dtype=torch.float32)
    return [x]


def _run_torch_ref(inputs):
    with torch.no_grad():
        return _model("PyTorch / ACL")(*inputs)


def _run_provider(key, inputs):
    with torch.no_grad():
        return _model(key)(*inputs)


def _flatten(o):
    if isinstance(o, (tuple, list)):
        return [x for v in o for x in _flatten(v)]
    return [o]


def _allclose(ref, out):
    refs = _flatten(ref)
    outs = _flatten(out)
    if len(refs) != len(outs):
        return False, "output_count_mismatch"
    for idx, (a, b) in enumerate(zip(refs, outs)):
        if a.shape != b.shape:
            return False, f"shape_mismatch[{idx}] {tuple(a.shape)} vs {tuple(b.shape)}"
        if not torch.allclose(a, b, rtol=1e-3, atol=1e-3):
            diff = (a - b).abs()
            max_abs = diff.max().item() if diff.numel() else 0.0
            denom = a.abs().clamp_min(1e-6)
            max_rel = (diff / denom).max().item() if diff.numel() else 0.0
            return False, f"tensor[{idx}] max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
    return True, "ok"


def _unsafe_baseline_grid(key, shape):
    if key == "Optimized Triton":
        return False
    n_tiles = triton.cdiv(math.prod(shape), 4096)
    return n_tiles > 65535


def unit_test():
    print("[UNIT_TEST] start")
    opt_ok = True
    for label, shape in _BENCH_SHAPES:
        inputs = _make_inputs(shape)
        ref = _run_torch_ref(inputs)
        torch.npu.synchronize()
        for key in _PROVIDER_FILES:
            if _unsafe_baseline_grid(key, shape):
                print(
                    f"[UNIT_TEST] {label} {key}: INFO_SKIP coreDim would exceed 65535"
                )
                continue
            try:
                out = _run_provider(key, inputs)
                torch.npu.synchronize()
                ok, msg = _allclose(ref, out)
                status = "PASS" if ok else "MISMATCH"
                print(f"[UNIT_TEST] {label} {key}: {status} {msg}")
                if key == "Optimized Triton" and not ok:
                    opt_ok = False
            except Exception as exc:
                torch.npu.synchronize()
                print(
                    f"[UNIT_TEST] {label} {key}: INFO_EXCEPTION {type(exc).__name__}: {exc}"
                )
                if key == "Optimized Triton":
                    opt_ok = False
        del inputs, ref
        torch.npu.empty_cache()
    print("UNIT_TEST PASS" if opt_ok else "UNIT_TEST_FAILED")
    return opt_ok


def _bench_callable(provider, shape):
    if provider != "PyTorch / ACL" and _unsafe_baseline_grid(provider, shape):
        print(
            f"[BENCH] {provider} {shape}: INFO_SKIP coreDim would exceed 65535"
        )
        return None
    inputs = _make_inputs(shape)
    if provider == "PyTorch / ACL":

        def fn():
            return _run_torch_ref(inputs)
    else:

        def fn():
            return _run_provider(provider, inputs)

    # Warm once so compile/autotune is outside timing.
    try:
        fn()
        torch.npu.synchronize()
    except Exception as exc:
        print(
            f"[BENCH] {provider} {shape}: INFO_EXCEPTION {type(exc).__name__}: {exc}"
        )
        return None
    return fn


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("black", "-"), ("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="Latency (ms)",
        plot_name="swish-performance",
        args={},
    ))
def bench(label, provider):
    shape = dict((name, shp) for name, shp in _BENCH_SHAPES)[label]
    fn = _bench_callable(provider, shape)
    if fn is None:
        return float("inf")
    try:
        return triton.testing.do_bench(fn,
                                       warmup=5,
                                       rep=20,
                                       return_mode="mean")
    except Exception:
        times = []
        for _ in range(10):
            fn()
            torch.npu.synchronize()
        for _ in range(50):
            start = time.perf_counter()
            fn()
            torch.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
        return sum(times) / len(times)


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
        bench.run(print_data=True,
                  show_plots=False,
                  save_path=str(ROOT / "profile_plots"))


if __name__ == "__main__":
    main()
