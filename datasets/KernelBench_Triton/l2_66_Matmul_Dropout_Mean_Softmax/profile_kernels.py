import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "66_Matmul_Dropout_Mean_Softmax.py"
OPT_FILE = ROOT / "opt_66_Matmul_Dropout_Mean_Softmax.py"
BASE2_FILE = ROOT / "base_66_Matmul_Dropout_Mean_Softmax.py"  # intentionally not imported in this sandbox

_BENCH_SHAPES = [
    ("tiny", 17, 100),
    ("default", 128, 100),
    ("large_direct", 4096, 100),
    ("persistent_forced", 2049, 100),
]

_MODULES = {}
_MODELS = {}


def _load(path, key):
    if key in _MODULES:
        return _MODULES[key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULES[key] = mod
    return mod


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    return init


def _model(key):
    if key in _MODELS:
        return _MODELS[key]
    if key == "baseline1":
        mod = _load(INPUT_FILE, key)
    elif key == "optimized":
        mod = _load(OPT_FILE, key)
    else:
        raise KeyError(key)
    model = mod.ModelNew(*_init_args(mod)).npu().eval()
    _MODELS[key] = model
    return model


def _make_inputs(label, batch, in_features):
    torch.manual_seed(0)
    return [torch.randn(batch, in_features, device="npu", dtype=torch.float32)]


def _run_torch_ref(x):
    return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)


def _run_provider(provider, x):
    if provider == "torch":
        return _run_torch_ref(x)
    if provider == "baseline2":
        raise RuntimeError(
            "Baseline Triton2 is read-only and unavailable in this sandbox")
    return _model(provider)(x)


def _max_abs(a, b):
    return float(
        (a - b).abs().max().detach().cpu().item()) if a.numel() else 0.0


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _bench_fn(fn, warmup=25, rep=100):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(5):
            fn()
        _sync()
        start = time.perf_counter()
        for _ in range(rep):
            fn()
        _sync()
        return (time.perf_counter() - start) * 1000.0 / rep


def unit_test():
    ok = True
    opt_mod = _load(OPT_FILE, "optimized")
    saved_max = getattr(opt_mod, "_MAX_PROGRAMS", None)
    for label, batch, in_features in _BENCH_SHAPES:
        x, = _make_inputs(label, batch, in_features)
        ref = _run_torch_ref(x)
        for provider, display in [
            ("baseline1", "Baseline Triton1"),
            ("baseline2", "Baseline Triton2"),
            ("optimized", "Optimized Triton"),
        ]:
            if provider == "baseline2":
                print(
                    f"TEST {display} {label}: SKIP_UNAVAILABLE sandbox_do_not_read_base max_abs=inf"
                )
                continue
            try:
                if provider == "optimized" and label == "persistent_forced" and saved_max is not None:
                    opt_mod._MAX_PROGRAMS = 1
                    _MODELS.pop("optimized", None)
                y = _run_provider(provider, x)
                _sync()
                diff = _max_abs(y, ref)
                passed = diff <= 1e-6 and y.shape == ref.shape
                ok = ok and (passed or provider == "baseline1")
                print(
                    f"TEST {display} {label}: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}"
                )
            except Exception as exc:
                ok = ok and provider != "optimized"
                print(
                    f"TEST {display} {label}: SKIP_UNAVAILABLE {type(exc).__name__} max_abs=inf"
                )
            finally:
                if provider == "optimized" and saved_max is not None:
                    opt_mod._MAX_PROGRAMS = saved_max
                    _MODELS.pop("optimized", None)
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


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
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="matmul_dropout_mean_softmax_latency",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    x, = _make_inputs(*shape)
    if provider == "baseline2":
        return float("inf")
    try:
        y_ref = _run_torch_ref(x)
        y = _run_provider(provider, x)
        _sync()
        if _max_abs(y, y_ref) > 1e-6:
            return float("inf")
        return _bench_fn(lambda: _run_provider(provider, x))
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
        benchmark.run(print_data=True,
                      show_plots=False,
                      save_path=str(ROOT / "profile_plots"))


if __name__ == "__main__":
    main()
