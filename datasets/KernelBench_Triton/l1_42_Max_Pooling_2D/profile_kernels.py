import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

import torch_npu  # noqa: F401

ROOT = Path(__file__).resolve().parent
_FILES = {
    "baseline1": "42_Max_Pooling_2D.py",
    "baseline2": "base_42_Max_Pooling_2D.py",
    "optimized": "opt_42_Max_Pooling_2D.py",
}
_MODULES = {}
_MODELS = {}

# label, N, C, H, W, kernel, stride, padding, dilation
_BENCH_SHAPES = [
    ("direct_small", 1, 2, 32, 32, 4, 1, 1, 1),
    ("generic_2x2", 1, 2, 32, 32, 2, 2, 1, 1),
    ("persistent_medium", 256, 64, 32, 32, 4, 1, 1, 1),
    ("target_original", 32, 64, 512, 512, 4, 1, 1, 1),
]
_LABEL_TO_SHAPE = {row[0]: row for row in _BENCH_SHAPES}


def _load(key):
    if key in _MODULES:
        return _MODULES[key]
    path = ROOT / _FILES[key]
    name = f"k_{key}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULES[key] = mod
    return mod


def _model(key, k, s, p, d):
    cache_key = (key, k, s, p, d)
    if cache_key not in _MODELS:
        mod = _load(key)
        _MODELS[cache_key] = mod.ModelNew(k, s, p, d).to(device="npu").eval()
    return _MODELS[cache_key]


def _make_input(label):
    _, n, c, h, w, *_ = _LABEL_TO_SHAPE[label]
    torch.manual_seed(0)
    return torch.rand((n, c, h, w), device="npu", dtype=torch.float32)


def _torch_ref(x, k, s, p, d):
    return F.max_pool2d(x, kernel_size=k, stride=s, padding=p, dilation=d)


def _run_provider(key, x, k, s, p, d):
    return _model(key, k, s, p, d)(x)


def _close(a, b):
    if a.shape != b.shape:
        return False, float("inf"), float("inf")
    diff = (a - b).abs()
    max_abs = float(diff.max().detach().cpu()) if diff.numel() else 0.0
    denom = b.abs().clamp_min(1e-6)
    max_rel = float(
        (diff / denom).max().detach().cpu()) if diff.numel() else 0.0
    return bool(torch.allclose(a, b, rtol=1e-3, atol=1e-3)), max_abs, max_rel


def unit_test():
    ok_opt = True
    for label, _, _, _, _, k, s, p, d in _BENCH_SHAPES:
        x = _make_input(label)
        ref = _torch_ref(x, k, s, p, d)
        for key in ("baseline1", "baseline2", "optimized"):
            try:
                out = _run_provider(key, x, k, s, p, d)
                torch.npu.synchronize()
                ok, ma, mr = _close(out, ref)
                status = "PASS" if ok else f"MISMATCH max_abs={ma:.6g} max_rel={mr:.6g}"
                print(f"TEST {key} {label} {status}")
                if key == "optimized" and not ok:
                    ok_opt = False
            except Exception as e:
                if key == "optimized":
                    print(
                        f"TEST {key} {label} EXCEPTION {type(e).__name__}: {e}"
                    )
                    ok_opt = False
                else:
                    print(
                        f"TEST {key} {label} INFO_EXCEPTION {type(e).__name__}"
                    )
    print("UNIT_TEST PASS" if ok_opt else "UNIT_TEST_FAILED")
    return ok_opt


def _time_fn(fn, warmup=2, rep=5):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
            torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(rep):
            fn()
            torch.npu.synchronize()
        return (time.perf_counter() - start) * 1000.0 / rep


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[row[0] for row in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "baseline1", "baseline2", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("green", "-"), ("black", "-"), ("red", "-")],
        ylabel="ms",
        plot_name="max_pool2d_perf",
        args={},
    ))
def bench(label, provider):
    _, _, _, _, _, k, s, p, d = _LABEL_TO_SHAPE[label]
    x = _make_input(label)
    if provider == "torch":

        def fn():
            return _torch_ref(x, k, s, p, d)
    else:
        try:
            # Avoid poisoning on known baseline grid overflow; keep the comparison column as inf.
            if provider in ("baseline1",
                            "baseline2") and label in ("persistent_medium",
                                                       "target_original"):
                return float("inf")
            model = _model(provider, k, s, p, d)

            def fn():
                return model(x)
        except Exception as e:
            print(f"INFO bench_setup {provider} {label} {type(e).__name__}")
            return float("inf")
    try:
        return _time_fn(fn)
    except Exception as e:
        print(f"INFO bench_exception {provider} {label} {type(e).__name__}")
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
