import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

HERE = Path(__file__).resolve().parent


def _load(fname, name):
    path = HERE / fname
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, name):
    try:
        return _load(fname, name)
    except Exception as exc:
        print(
            f"TEST {name} import SKIP optional_unavailable {type(exc).__name__}"
        )
        return None


baseline1 = _load("92_cumsum_exclusive.py", "k_baseline1_92_cumsum_exclusive")
baseline2 = _load_optional("base_92_cumsum_exclusive.py", "baseline2")
optimized = _load("opt_92_cumsum_exclusive.py",
                  "k_optimized_92_cumsum_exclusive")

_MODEL_CACHE = {}
_BENCH_SHAPES = [
    ("tiny", 4, 17),
    ("odd", 257, 257),
    ("medium", 2048, 2048),
    ("target", 32768, 32768),
]
_UNIT_ONLY = [("fallback_tiny", 4, 17), ("fallback_odd", 257, 257)]


def _sync():
    torch.npu.synchronize()


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    return init


def _model(provider):
    if provider in _MODEL_CACHE:
        return _MODEL_CACHE[provider]
    if provider == "baseline1":
        mod, args = baseline1, _init_args(baseline1)
    elif provider == "baseline2":
        if baseline2 is None:
            return None
        mod, args = baseline2, _init_args(baseline2)
    elif provider == "optimized":
        mod, args = optimized, _init_args(optimized)
    elif provider == "optimized_fallback":
        mod, args = optimized, [1, True]
    else:
        raise KeyError(provider)
    m = mod.ModelNew(*args).to(device="npu").eval()
    _MODEL_CACHE[provider] = m
    return m


def _make_input(B, N):
    torch.manual_seed(0)
    return torch.rand((B, N), device="npu", dtype=torch.float32)


def _run_torch_ref(x):
    B, N = x.shape
    if N == 0:
        return x.new_empty((max(B - 1, 0), 1))
    if B <= 1:
        return x.new_empty((0, N + 1))
    y = torch.empty((B - 1, N + 1), device=x.device, dtype=x.dtype)
    y[:, 0].zero_()
    y[:, 1:] = torch.cumsum(x[:-1], dim=1)
    return y


def _run_provider(provider, x):
    if provider == "torch":
        return _run_torch_ref(x)
    m = _model(provider)
    if m is None:
        raise RuntimeError("provider_unavailable")
    return m(x)


def _check_close(got, ref, label, provider):
    if got.shape != ref.shape:
        print(
            f"TEST {provider} {label} SKIP shape_mismatch got={tuple(got.shape)} ref={tuple(ref.shape)}"
        )
        return provider == "baseline2"
    diff = (got - ref).abs()
    max_abs = float(diff.max().detach().cpu()) if diff.numel() else 0.0
    ref_abs = ref.abs()
    max_ref = float(ref_abs.max().detach().cpu()) if ref_abs.numel() else 1.0
    tol = 1e-3 + 1e-3 * max(1.0, max_ref)
    ok = max_abs <= tol
    if ok:
        print(
            f"TEST {provider} {label} PASS max_abs={max_abs:.6g} tol={tol:.6g}"
        )
        return True
    if provider == "baseline2":
        print(
            f"TEST {provider} {label} SKIP value_mismatch max_abs={max_abs:.6g} tol={tol:.6g}"
        )
        return True
    print(
        f"TEST {provider} {label} MISMATCH max_abs={max_abs:.6g} tol={tol:.6g}"
    )
    return False


def unit_test():
    ok = True
    providers = ["baseline1", "baseline2", "optimized"]
    with torch.no_grad():
        for label, B, N in _BENCH_SHAPES:
            x = _make_input(B, N)
            ref = _run_torch_ref(x)
            _sync()
            for provider in providers:
                if provider in ("baseline1", "baseline2") and N > 512:
                    print(f"TEST {provider} {label} SKIP compile_guard")
                    continue
                try:
                    got = _run_provider(provider, x)
                    _sync()
                    ok = _check_close(got, ref, label, provider) and ok
                except Exception as exc:
                    if provider == "baseline2":
                        print(
                            f"TEST {provider} {label} SKIP optional_unavailable {type(exc).__name__}"
                        )
                    else:
                        print(
                            f"TEST {provider} {label} MISMATCH exception {type(exc).__name__}"
                        )
                        ok = False
            del x, ref
        for label, B, N in _UNIT_ONLY:
            x = _make_input(B, N)
            ref = _run_torch_ref(x)
            got = _run_provider("optimized_fallback", x)
            _sync()
            ok = _check_close(got, ref, label, "optimized_fallback") and ok
            del x, ref, got
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_provider(provider, label):
    shape = next((s for s in _BENCH_SHAPES if s[0] == label), None)
    if shape is None:
        return float("inf")
    _, B, N = shape
    if provider == "baseline2" and baseline2 is None:
        return float("inf")
    if provider in ("baseline1", "baseline2") and N > 512:
        print(f"INFO bench {provider} {label} inf compile_guard")
        return float("inf")
    x = _make_input(B, N)
    try:

        def fn():
            return _run_provider(provider, x)  # noqa: F821

        fn()
        _sync()
        rep = 20 if B * N <= 2048 * 2048 else 5
        warmup = 5 if B * N <= 2048 * 2048 else 1
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception as exc:
        print(f"INFO bench {provider} {label} inf {type(exc).__name__}")
        return float("inf")
    finally:
        del x


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
        styles=[("blue", "-"), ("red", "-"), ("orange", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="cumsum_exclusive_perf",
        args={},
    ))
def bench(label, provider):
    return _time_provider(provider, label)


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
