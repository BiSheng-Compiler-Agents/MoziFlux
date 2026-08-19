import argparse
import importlib.util
import pathlib
import sys
import time

import torch
import triton

ROOT = pathlib.Path(__file__).resolve().parent
INPUT_FILE = ROOT / "80_Gemm_Max_Subtract_GELU.py"
OPT_FILE = ROOT / "opt_80_Gemm_Max_Subtract_GELU.py"

_BENCH_SHAPES = [
    ("small", 32, 512, 1024, 1),
    ("irregular", 257, 768, 1536, 1),
    ("target", 1024, 8192, 8192, 1),
]

_MODEL_CACHE = {}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _provider_mod(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if key == "baseline1":
        mod = _load(INPUT_FILE, "k_l2_80_baseline1")
    elif key == "optimized":
        mod = _load(OPT_FILE, "k_l2_80_optimized")
    else:
        mod = None
    _MODEL_CACHE[key] = mod
    return mod


def _model(key, in_features, out_features, max_dim):
    cache_key = (key, in_features, out_features, max_dim)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    mod = _provider_mod(key)
    torch.manual_seed(0)
    model = mod.ModelNew(in_features, out_features, max_dim).npu().eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _make_inputs(batch, in_features, dtype=torch.float16):
    torch.manual_seed(123)
    return torch.rand((batch, in_features), device="npu", dtype=dtype)


def _run_torch_ref(x, out_features, max_dim):
    # Algebraic contract from the source kernel: max tensor is subtracted from itself,
    # then GELU(0) is zero. Avoids spending the reference path on a dead GEMM.
    if max_dim < 0:
        max_dim = 2 + max_dim
    shape = list(x.shape)
    if max_dim == 1:
        shape[1] = 1
    elif max_dim == 0:
        shape[0] = 1
    else:
        shape = [x.shape[0], 1]
    return torch.zeros(tuple(shape), device=x.device, dtype=x.dtype)


def _run_provider(key, x, in_features, out_features, max_dim):
    if key == "torch":
        return _run_torch_ref(x, out_features, max_dim)
    if key == "baseline2":
        raise RuntimeError("sandbox_reference_file_not_read")
    model = _model(key, in_features, out_features, max_dim)
    return model(x)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _time_call(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


def _bench_one(provider, label):
    row = next(r for r in _BENCH_SHAPES if r[0] == label)
    _, batch, in_features, out_features, max_dim = row
    if provider == "baseline2":
        print(
            f"INFO benchmark Baseline Triton2 {label}: unavailable_reference_sandbox -> inf"
        )
        return float("inf")
    x = _make_inputs(batch, in_features)

    def fn():
        with torch.no_grad():
            return _run_provider(provider, x, in_features, out_features,
                                 max_dim)

    try:
        if hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn,
                                           warmup=10,
                                           rep=50,
                                           return_mode="mean")
        return _time_call(fn)
    except Exception as e:
        print(f"INFO benchmark {provider} {label}: {type(e).__name__} -> inf")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[r[0] for r in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "baseline1", "baseline2", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="l2_80_gemm_max_subtract_gelu",
        args={},
    ))
def bench(label, provider):
    return _bench_one(provider, label)


def unit_test():
    ok = True
    for label, batch, in_features, out_features, max_dim in _BENCH_SHAPES:
        x = _make_inputs(batch, in_features)
        ref = _run_torch_ref(x, out_features, max_dim)
        for provider, display in [
            ("torch", "PyTorch / ACL"),
            ("baseline1", "Baseline Triton1"),
            ("baseline2", "Baseline Triton2"),
            ("optimized", "Optimized Triton"),
        ]:
            if provider == "baseline2":
                print(
                    f"TEST {display} {label}: SKIP_REFERENCE_SANDBOX max_abs=inf"
                )
                continue
            try:
                with torch.no_grad():
                    out = _run_provider(provider, x, in_features, out_features,
                                        max_dim)
                max_abs = (out -
                           ref).abs().max().item() if out.numel() else 0.0
                passed = out.shape == ref.shape and max_abs <= 1e-3
                print(
                    f"TEST {display} {label}: {'PASS' if passed else 'FAIL'} max_abs={max_abs:.6g}"
                )
                ok = ok and passed
            except Exception as e:
                print(
                    f"TEST {display} {label}: FAIL {type(e).__name__} max_abs=inf"
                )
                if provider == "optimized":
                    ok = False
        if label == "small":
            opt = _provider_mod("optimized")
            old = opt._MAX_PROGRAMS
            try:
                opt._MAX_PROGRAMS = 1
                _MODEL_CACHE.pop(
                    ("optimized", in_features, out_features, max_dim), None)
                out = _run_provider("optimized", x, in_features, out_features,
                                    max_dim)
                max_abs = (out -
                           ref).abs().max().item() if out.numel() else 0.0
                passed = out.shape == ref.shape and max_abs <= 1e-3
                print(
                    f"TEST Optimized Triton forced_persistent_{label}: {'PASS' if passed else 'FAIL'} max_abs={max_abs:.6g}"
                )
                ok = ok and passed
            finally:
                opt._MAX_PROGRAMS = old
                _MODEL_CACHE.pop(
                    ("optimized", in_features, out_features, max_dim), None)
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


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
