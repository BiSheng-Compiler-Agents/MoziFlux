import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent


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
        print(f"INFO {key} import_skip {type(exc).__name__}")
        return None


baseline1_mod = _load("97_CosineSimilarityLoss.py", "baseline1")
baseline2_mod = _load_optional("base_97_CosineSimilarityLoss.py", "baseline2")
optimized_mod = _load("opt_97_CosineSimilarityLoss.py", "optimized")

_BENCH_SHAPES = [
    ("small_4x128", 4, 128),
    ("irregular_129x1000", 129, 1000),
    ("exact_128x4096", 128, 4096),
]
_MODEL_CACHE = {}


def _sync():
    torch.npu.synchronize()


def _init_args(mod):
    if mod is None or not hasattr(mod, "get_init_inputs"):
        return []
    init = mod.get_init_inputs()
    if init == [()]:
        return []
    return init


def _model(provider, use_fallback=False):
    key = (provider, use_fallback)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    mod = {
        "baseline1": baseline1_mod,
        "baseline2": baseline2_mod,
        "optimized": optimized_mod
    }[provider]
    if mod is None:
        return None
    model = mod.ModelNew(*_init_args(mod))
    model = model.to(device="npu").eval()
    _MODEL_CACHE[key] = model
    return model


def _make_inputs(B, D):
    torch.manual_seed(0)
    x = torch.randn((B, D), device="npu", dtype=torch.float32)
    y = torch.randn((B, D), device="npu", dtype=torch.float32)
    return x, y


def _run_torch_ref(x, y):
    return (1.0 - F.cosine_similarity(
        x.to(torch.float32), y.to(torch.float32), dim=1, eps=1e-8)).mean()


def _grid_guard(provider, B):
    if provider in ("baseline1", "baseline2") and B > 65535:
        return False
    return True


def _run_provider(provider, x, y, use_fallback=False):
    if provider == "torch":
        return _run_torch_ref(x, y)
    if provider == "baseline2" and baseline2_mod is None:
        raise RuntimeError("baseline2_unavailable")
    if not _grid_guard(provider, x.shape[0]):
        raise RuntimeError("grid_guard")
    model = _model(provider, use_fallback=use_fallback)
    return model(x, y)


def _max_diff(a, b):
    return float(
        (a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def unit_test():
    ok = True
    for label, B, D in _BENCH_SHAPES:
        x, y = _make_inputs(B, D)
        ref = _run_torch_ref(x, y)
        _sync()
        for provider in ("baseline1", "baseline2", "optimized"):
            if provider == "baseline2" and baseline2_mod is None:
                print(f"TEST {provider} {label} SKIP unavailable")
                continue
            try:
                out = _run_provider(provider, x, y)
                _sync()
                diff = _max_diff(out, ref)
                if diff <= 1e-4:
                    print(f"TEST {provider} {label} PASS max_diff={diff:.6g}")
                elif provider == "optimized":
                    print(f"TEST {provider} {label} BAD max_diff={diff:.6g}")
                    ok = False
                else:
                    print(
                        f"TEST {provider} {label} SKIP value_diff={diff:.6g}")
            except Exception as exc:
                if provider == "optimized":
                    print(
                        f"TEST {provider} {label} BAD exception={type(exc).__name__}"
                    )
                    ok = False
                else:
                    print(f"TEST {provider} {label} SKIP {type(exc).__name__}")
    # Dispatch-only test for optimized persistent fallback: large B, tiny D keeps memory bounded.
    label, B, D = "persistent_fallback_65536x1", 65536, 1
    x, y = _make_inputs(B, D)
    ref = _run_torch_ref(x, y)
    try:
        out = _run_provider("optimized", x, y, use_fallback=True)
        _sync()
        diff = _max_diff(out, ref)
        if diff <= 1e-4:
            print(f"TEST optimized_fallback {label} PASS max_diff={diff:.6g}")
        else:
            print(f"TEST optimized_fallback {label} BAD max_diff={diff:.6g}")
            ok = False
    except Exception as exc:
        print(
            f"TEST optimized_fallback {label} BAD exception={type(exc).__name__}"
        )
        ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_manual(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
        _sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
        _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


def _bench_fn(fn):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=25,
                                       rep=100,
                                       return_mode="mean")
    except Exception:
        return _bench_manual(fn)


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
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "-")],
        ylabel="ms",
        plot_name="cosine_similarity_loss",
        args={},
    ))
def benchmark(label, provider):
    shape = {s[0]: s[1:] for s in _BENCH_SHAPES}[label]
    B, D = shape
    if provider == "baseline2" and baseline2_mod is None:
        return float("inf")
    if not _grid_guard(provider, B):
        return float("inf")
    x, y = _make_inputs(B, D)
    try:
        out = _run_provider(provider, x, y)
        _sync()
        ref = _run_torch_ref(x, y)
        _sync()
        if _max_diff(out, ref) > 1e-4 and provider == "optimized":
            return float("inf")
        return _bench_fn(lambda: _run_provider(provider, x, y))
    except Exception as exc:
        print(f"INFO bench_{provider}_{label} skip {type(exc).__name__}")
        return float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not args.test and not args.bench:
        args.test = True
        args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
