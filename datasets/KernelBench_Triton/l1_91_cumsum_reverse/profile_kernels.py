import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, ROOT / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, modname):
    try:
        return _load(fname, modname)
    except Exception as exc:
        print(f"INFO {modname} import_skip {type(exc).__name__}")
        return None


_baseline1 = _load("91_cumsum_reverse.py", "k_baseline1_91_cumsum_reverse")
_baseline2 = _load_optional("base_91_cumsum_reverse.py",
                            "k_baseline2_91_cumsum_reverse")
_optimized = _load("opt_91_cumsum_reverse.py", "k_optimized_91_cumsum_reverse")

_PROVIDERS = [
    ("baseline1", "Baseline Triton1", _baseline1),
    ("baseline2", "Baseline Triton2", _baseline2),
    ("optimized", "Optimized Triton", _optimized),
]

_MODEL_CACHE = {}
_BENCH_SHAPES = [
    ("tiny", 8, 64),
    ("mask_odd", 7, 1000),
    ("medium", 128, 4096),
    ("target", 32768, 32768),
]


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    return list(init)


def _model(key, mod):
    cache_key = key
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = mod.ModelNew(*_init_args(mod)).to(
            device="npu").eval()
    return _MODEL_CACHE[cache_key]


def _make_input(M, N):
    torch.manual_seed(0)
    # Source get_inputs uses torch.rand() default fp32 on CPU.
    return torch.rand((M, N), device="npu", dtype=torch.float32)


def _torch_ref(x, dim=1):
    return torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim),
                      dims=[dim])


def _run_provider(key, mod, x):
    if mod is None:
        raise RuntimeError("provider_unavailable")
    with torch.no_grad():
        return _model(key, mod)(x)


def _sync():
    torch.npu.synchronize()


def _close(a, b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-3)


def unit_test():
    ok_opt = True
    for label, M, N in _BENCH_SHAPES:
        x = _make_input(M, N)
        ref = _torch_ref(x)
        _sync()
        for key, _name, mod in _PROVIDERS:
            if mod is None:
                print(f"TEST {key} {label} SKIP provider_unavailable")
                continue
            try:
                out = _run_provider(key, mod, x)
                _sync()
                same = _close(out, ref)
                if same:
                    print(f"TEST {key} {label} PASS")
                else:
                    max_err = (out - ref).abs().max().item()
                    if key == "optimized":
                        ok_opt = False
                        print(
                            f"TEST {key} {label} MISMATCH max_err={max_err:.6g}"
                        )
                    else:
                        print(
                            f"TEST {key} {label} SKIP value_mismatch max_err={max_err:.6g}"
                        )
            except Exception as exc:
                if key == "optimized":
                    ok_opt = False
                    print(
                        f"TEST {key} {label} MISMATCH exception={type(exc).__name__}"
                    )
                else:
                    print(
                        f"TEST {key} {label} SKIP provider_exception_{type(exc).__name__}"
                    )
        del x, ref
    print("UNIT_TEST PASS" if ok_opt else "UNIT_TEST_FAILED")
    return ok_opt


def _bench_one(fn, warmup=5, rep=20):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(3):
            fn()
            _sync()
        start = time.perf_counter()
        for _ in range(10):
            fn()
            _sync()
        return (time.perf_counter() - start) * 1000.0 / 10.0


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
        styles=[("blue", "-"), ("red", "-"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="cumsum_reverse_latency",
        args={},
    ))
def bench(label, provider):
    shape = {s[0]: s[1:] for s in _BENCH_SHAPES}[label]
    x = _make_input(*shape)
    if provider == "torch":

        def fn():
            return _torch_ref(x)
    else:
        mod = {k: m for k, _, m in _PROVIDERS}[provider]
        if mod is None:
            print(f"INFO bench {provider} {label} inf provider_unavailable")
            return float("inf")

        def fn():
            return _run_provider(provider, mod, x)

    try:
        ms = _bench_one(fn)
        return ms
    except Exception as exc:
        print(f"INFO bench {provider} {label} inf {type(exc).__name__}")
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
