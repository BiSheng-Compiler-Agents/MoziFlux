import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "38_L1Norm_.py"
BASE_FILE = HERE / "base_38_L1Norm_.py"
OPT_FILE = HERE / "opt_38_L1Norm_.py"

_BENCH_SHAPES = [
    ("tiny_block64", 3, 65),
    ("small_block1024", 4, 2048),
    ("mid_block2048", 8, 8192),
    ("large_block4096", 8, 16384),
    ("target", 32768, 65535),
]

_MODULE_CACHE = {}
_MODEL_CACHE = {}
_PROVIDERS = [("baseline1", INPUT_FILE), ("optimized", OPT_FILE)]
if BASE_FILE.exists():
    _PROVIDERS.insert(1, ("baseline2", BASE_FILE))


def _load(key, path):
    if key not in _MODULE_CACHE:
        spec = importlib.util.spec_from_file_location(
            f"k_l1norm_{key}_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MODULE_CACHE[key] = mod
    return _MODULE_CACHE[key]


def _model(key):
    if key not in _MODEL_CACHE:
        path = dict(_PROVIDERS)[key]
        mod = _load(key, path)
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        if init == [()]:
            init = []
        _MODEL_CACHE[key] = mod.ModelNew(*init).to(device="npu").eval()
    return _MODEL_CACHE[key]


def _make_inputs(label):
    shapes = {name: (b, n) for name, b, n in _BENCH_SHAPES}
    b, n = shapes[label]
    torch.manual_seed(0)
    x = torch.rand((b, n), device="npu", dtype=torch.float32)
    return [x]


def _run_torch_ref(x):
    denom = torch.sum(torch.abs(x), dim=1, keepdim=True)
    return x / denom


def _run_provider(key, x):
    return _model(key)(x)


def _sync():
    torch.npu.synchronize()


def _close(a, b):
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return bool(
            torch.equal(torch.isfinite(a), torch.isfinite(b))
            and torch.allclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=True))
    return bool(torch.allclose(a, b, rtol=1e-4, atol=1e-5))


def unit_test():
    ok = True
    for label, _, _ in _BENCH_SHAPES:
        x, = _make_inputs(label)
        try:
            ref = _run_torch_ref(x)
            _sync()
        except Exception as e:
            print(f"TEST torch_ref {label} ERROR {type(e).__name__}: {e}")
            ok = False
            continue
        for key, _ in _PROVIDERS:
            try:
                y = _run_provider(key, x)
                _sync()
                if _close(y, ref):
                    print(f"TEST {key} {label} PASS")
                else:
                    max_err = (y - ref).abs().max().item()
                    tag = "MISMATCH" if key == "optimized" else "INFO_MISMATCH"
                    print(f"TEST {key} {label} {tag} max_abs={max_err:.6e}")
                    if key == "optimized":
                        ok = False
            except Exception as e:
                tag = "ERROR" if key == "optimized" else "INFO_ERROR"
                print(f"TEST {key} {label} {tag} {type(e).__name__}: {e}")
                if key == "optimized":
                    ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_callable(provider, label):
    x, = _make_inputs(label)
    if provider == "torch":

        def fn():
            return _run_torch_ref(x)
    else:

        def fn():
            return _run_provider(provider, x)

    # Compile/warmup outside timing and keep target reps intentionally low.
    warmup = 1 if label == "target" else 5
    rep = 3 if label == "target" else 20
    for _ in range(warmup):
        fn()
        _sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
        _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


_line_names = ["PyTorch / ACL"]
_line_vals = ["torch"]
_line_styles = [("black", "-")]
if "baseline1" in dict(_PROVIDERS):
    _line_names.append(
        "Baseline Triton1" if BASE_FILE.exists() else "Baseline Triton")
    _line_vals.append("baseline1")
    _line_styles.append(("blue", "-"))
if "baseline2" in dict(_PROVIDERS):
    _line_names.append("Baseline Triton2")
    _line_vals.append("baseline2")
    _line_styles.append(("red", "--"))
_line_names.append("Optimized Triton")
_line_vals.append("optimized")
_line_styles.append(("green", "-"))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_line_vals,
        line_names=_line_names,
        styles=_line_styles,
        ylabel="ms",
        plot_name="l1norm-performance",
        args={},
    ))
def benchmark(label, provider):
    try:
        return _bench_callable(provider, label)
    except Exception as e:
        print(f"INFO bench {provider} {label} inf {type(e).__name__}: {e}")
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
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
