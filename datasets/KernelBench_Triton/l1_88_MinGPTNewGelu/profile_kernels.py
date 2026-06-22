import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent


def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {fname}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, name):
    try:
        return _load(fname, name)
    except Exception as exc:
        print(f"INFO {name} import_unavailable {type(exc).__name__}")
        return None


_MODS = {
    "baseline1":
    _load("88_MinGPTNewGelu.py", "k_baseline1_88_MinGPTNewGelu"),
    "baseline2":
    _load_optional("base_88_MinGPTNewGelu.py", "k_baseline2_88_MinGPTNewGelu"),
    "optimized":
    _load("opt_88_MinGPTNewGelu.py", "k_optimized_88_MinGPTNewGelu"),
}
_MODEL_CACHE = {}

_BENCH_SHAPES = [
    ("odd_257x513", (257, 513)),
    ("medium_1024x2048", (1024, 2048)),
    ("target_8192x8192", (8192, 8192)),
]


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    return init


def _model(provider):
    if provider not in _MODEL_CACHE:
        mod = _MODS[provider]
        if mod is None:
            return None
        _MODEL_CACHE[provider] = mod.ModelNew(*_init_args(mod)).to(
            device="npu").eval()
    return _MODEL_CACHE[provider]


def _make_input(shape):
    torch.manual_seed(0)
    return torch.rand(shape, device="npu", dtype=torch.float32)


def _torch_ref(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 *
                                       (x + 0.044715 * x * x * x)))


def _run_provider(provider, x):
    if provider == "torch":
        return _torch_ref(x)
    model = _model(provider)
    if model is None:
        raise RuntimeError("provider_unavailable")
    with torch.no_grad():
        return model(x)


def _same(a, b):
    if a.shape != b.shape:
        return False, float("inf")
    diff = (a - b).abs()
    max_err = diff.max().item() if diff.numel() else 0.0
    ok = torch.allclose(a, b, rtol=1e-3, atol=1e-3)
    return bool(ok), float(max_err)


def unit_test():
    all_opt_ok = True
    for label, shape in _BENCH_SHAPES:
        x = _make_input(shape)
        ref = _run_provider("torch", x)
        for provider in ("baseline1", "baseline2", "optimized"):
            try:
                y = _run_provider(provider, x)
                torch.npu.synchronize()
                ok, max_err = _same(y, ref)
                if ok:
                    print(
                        f"TEST {provider} {label} PASS max_err={max_err:.6g}")
                else:
                    if provider == "optimized":
                        all_opt_ok = False
                        print(
                            f"TEST {provider} {label} MISMATCH max_err={max_err:.6g}"
                        )
                    else:
                        print(
                            f"TEST {provider} {label} SKIP value_mismatch max_err={max_err:.6g}"
                        )
            except Exception as exc:
                if provider == "optimized":
                    all_opt_ok = False
                    print(
                        f"TEST {provider} {label} MISMATCH exception={type(exc).__name__}"
                    )
                else:
                    print(
                        f"TEST {provider} {label} SKIP provider_unavailable {type(exc).__name__}"
                    )

    # Force optimized persistent dispatch on a small tensor so every dispatch path is tested
    opt_mod = _MODS["optimized"]
    old = getattr(opt_mod, "_MAX_PROGRAMS", 65535)
    try:
        opt_mod._MAX_PROGRAMS = 1
        _MODEL_CACHE.pop("optimized", None)
        x = _make_input((8, 2048))
        y = _run_provider("optimized", x)
        ref = _run_provider("torch", x)
        torch.npu.synchronize()
        ok, max_err = _same(y, ref)
        if ok:
            print(
                f"TEST optimized persistent_forced PASS max_err={max_err:.6g}")
        else:
            all_opt_ok = False
            print(
                f"TEST optimized persistent_forced MISMATCH max_err={max_err:.6g}"
            )
    finally:
        opt_mod._MAX_PROGRAMS = old
        _MODEL_CACHE.pop("optimized", None)

    print("UNIT_TEST PASS" if all_opt_ok else "UNIT_TEST_FAILED")
    return all_opt_ok


def _bench_callable(fn, warmup=5, rep=20):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
            torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end) / rep


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
        plot_name="mingpt_new_gelu",
        args={},
    ))
def benchmark(label, provider):
    shape = dict(_BENCH_SHAPES)[label]
    x = _make_input(shape)
    try:
        _ = _run_provider(provider, x)
        torch.npu.synchronize()
    except Exception as exc:
        print(f"INFO bench {provider} {label} inf {type(exc).__name__}")
        return float("inf")

    def fn():
        _run_provider(provider, x)

    try:
        return _bench_callable(fn)
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
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
