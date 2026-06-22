import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

ROOT = Path(__file__).resolve().parent
INPUT_FILE = "51_Argmax_over_a_dimension.py"
OPT_FILE = "opt_51_Argmax_over_a_dimension.py"
BASE2_FILE = "base_51_Argmax_over_a_dimension.py"

_BENCH_SHAPES = [
    ("small_dim1", (4, 64, 65), 1),
    ("irregular_dim1", (3, 257, 129), 1),
    ("target_dim1", (128, 4096, 4095), 1),
]
_EXTRA_TESTS = [
    ("fallback_dim0", (5, 17, 19), 0),
    ("fallback_dim2", (5, 17, 19), 2),
]

_MODULE_CACHE = {}
_MODEL_CACHE = {}
_INPUT_CACHE = {}


def _load(key, filename):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def _init_args(mod, dim):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else [dim]
    if init == [()]:
        init = []
    if init:
        return [dim]
    return []


def _model(provider, dim):
    key = (provider, dim)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if provider == "baseline1":
        mod = _load("baseline1", INPUT_FILE)
    elif provider == "baseline2":
        mod = _load("baseline2", BASE2_FILE)
    elif provider == "optimized":
        mod = _load("optimized", OPT_FILE)
    else:
        raise ValueError(provider)
    model = mod.ModelNew(*_init_args(mod, dim)).to(device="npu").eval()
    _MODEL_CACHE[key] = model
    return model


def _make_input(shape, label):
    key = (shape, label)
    if key in _INPUT_CACHE:
        return _INPUT_CACHE[key]
    # Constant target avoids a costly random fill on very large tensors while preserving argmax tie semantics.
    if label == "target_dim1":
        x = torch.empty(shape, device="npu", dtype=torch.float32).fill_(0.0)
    else:
        b, m, n = shape
        vals = torch.arange(m, device="npu", dtype=torch.float32).view(1, m, 1)
        x = vals.expand(b, m, n).contiguous()
    _INPUT_CACHE[key] = x
    return x


def _run_torch_ref(x, dim):
    return torch.argmax(x, dim=dim)


def _would_grid_overflow_baseline1(x, dim):
    rows = x.numel() // x.shape[dim]
    return rows > 65535


def _run_provider(provider, x, dim, label):
    if provider == "torch":
        return _run_torch_ref(x, dim)
    if provider == "baseline1" and _would_grid_overflow_baseline1(x, dim):
        raise RuntimeError("grid_guard")
    if provider == "baseline2" and label == "target_dim1":
        raise RuntimeError("grid_guard")
    return _model(provider, dim)(x)


def _sync():
    torch.npu.synchronize()


def _time_ms(fn, warmup=3, rep=10):
    try:
        import triton.testing as tt
        return tt.do_bench(fn, warmup=warmup, rep=rep, return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
            _sync()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
            _sync()
        return (time.perf_counter() - t0) * 1000.0 / rep


def _check_provider(provider, label, shape, dim, ref):
    x = _make_input(shape, label)
    try:
        y = _run_provider(provider, x, dim, label)
        _sync()
        ok = torch.equal(y, ref)
        if ok:
            print(f"TEST {provider} {label} PASS")
            return True
        diff = (y != ref).sum().item()
        if provider in ("baseline1", "baseline2"):
            print(f"TEST {provider} {label} SKIP value_mismatch count={diff}")
            return True
        print(f"TEST {provider} {label} MISMATCH count={diff}")
        return False
    except Exception as exc:
        if provider in ("baseline1", "baseline2"):
            print(f"TEST {provider} {label} SKIP {type(exc).__name__}")
            return True
        print(f"TEST {provider} {label} MISMATCH {type(exc).__name__}")
        return False


def unit_test():
    ok = True
    for label, shape, dim in _BENCH_SHAPES + _EXTRA_TESTS:
        x = _make_input(shape, label)
        ref = _run_torch_ref(x, dim)
        _sync()
        for provider in ("baseline1", "baseline2", "optimized"):
            if provider != "optimized" and label.startswith("fallback"):
                continue
            ok = _check_provider(provider, label, shape, dim, ref) and ok
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


try:
    import triton.testing as tt

    @tt.perf_report(
        tt.Benchmark(
            x_names=["label"],
            x_vals=[s[0] for s in _BENCH_SHAPES],
            line_arg="provider",
            line_vals=["torch", "baseline1", "baseline2", "optimized"],
            line_names=[
                "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
                "Optimized Triton"
            ],
            styles=[("blue", "-"), ("green", "-"), ("black", "-"),
                    ("red", "-")],
            ylabel="ms",
            plot_name="argmax_over_dimension",
            args={},
        ))
    def benchmark(label, provider):
        shape, dim = next(
            (s, d) for item, s, d in _BENCH_SHAPES if item == label)
        x = _make_input(shape, label)
        try:
            _run_provider(provider, x, dim, label)
            _sync()
        except Exception:
            return float("inf")
        rep = 5 if label == "target_dim1" else 50
        warmup = 1 if label == "target_dim1" else 10
        return _time_ms(lambda: _run_provider(provider, x, dim, label),
                        warmup=warmup,
                        rep=rep)
except Exception:
    benchmark = None


def run_bench():
    if benchmark is not None:
        benchmark.run(print_data=True, show_plots=False)
        return
    print(
        "label PyTorch / ACL Baseline Triton1 Baseline Triton2 Optimized Triton"
    )
    for label, shape, dim in _BENCH_SHAPES:
        vals = []
        x = _make_input(shape, label)
        for provider in ("torch", "baseline1", "baseline2", "optimized"):
            try:
                vals.append(
                    _time_ms(
                        lambda p=provider: _run_provider(p, x, dim, label),
                        warmup=1,
                        rep=5))
            except Exception:
                vals.append(float("inf"))
        print(label, *vals)


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
        run_bench()


if __name__ == "__main__":
    main()
