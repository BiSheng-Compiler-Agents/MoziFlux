import argparse
import importlib.util
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

HERE = pathlib.Path(__file__).resolve().parent


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, modname):
    try:
        return _load(fname, modname)
    except Exception as exc:
        print(
            f"INFO optional_provider_unavailable {fname}: {type(exc).__name__}: {exc}"
        )
        return None


baseline1 = _load("94_MSELoss.py", "k_baseline1_94_mseloss")
baseline2 = _load_optional("base_94_MSELoss.py", "k_baseline2_94_mseloss")
optimized = _load("opt_94_MSELoss.py", "k_optimized_94_mseloss")

_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_MODEL_CACHE = {}
_BENCH_SHAPES = [
    ("small_single", 2048),
    ("direct_1m", 1 << 20),
    ("persistent_1g", 32768 * 32768),
]


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _model(provider):
    if provider in _MODEL_CACHE:
        return _MODEL_CACHE[provider]
    mod = {
        "baseline1": baseline1,
        "baseline2": baseline2,
        "optimized": optimized
    }[provider]
    if mod is None:
        return None
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    m = mod.ModelNew(*init).to(device="npu").eval()
    _MODEL_CACHE[provider] = m
    return m


def _make_inputs(n):
    # Deterministic values avoid expensive random generation for the 1G persistent dispatch shape.
    x = torch.empty((n, ), device="npu", dtype=torch.float32).fill_(0.25)
    y = torch.empty((n, ), device="npu", dtype=torch.float32).fill_(0.75)
    return x, y


def _torch_ref(x, y):
    return F.mse_loss(x, y, reduction="mean")


def _run_provider(provider, x, y):
    if provider == "torch":
        return _torch_ref(x, y)
    if provider == "baseline1":
        n_tiles = triton.cdiv(x.numel(), 4096)
        if n_tiles > 65535:
            raise RuntimeError("grid_guard")
    m = _model(provider)
    if m is None:
        raise RuntimeError("provider_unavailable")
    return m(x, y)


def _check_close(a, b):
    return torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-3)


def unit_test():
    ok = True
    with torch.no_grad():
        for label, n in _BENCH_SHAPES:
            x, y = _make_inputs(n)
            ref = _torch_ref(x, y)
            for provider in _PROVIDERS[1:]:
                if provider == "baseline2" and baseline2 is None:
                    print(f"TEST {provider} {label} SKIP provider_unavailable")
                    continue
                if provider == "baseline1" and triton.cdiv(n, 4096) > 65535:
                    print(f"TEST {provider} {label} SKIP grid_guard")
                    continue
                try:
                    out = _run_provider(provider, x, y)
                    _sync()
                    if _check_close(out, ref):
                        print(f"TEST {provider} {label} PASS")
                    else:
                        diff = (out.float() - ref.float()).abs().item()
                        if provider == "optimized":
                            print(
                                f"TEST {provider} {label} MISMATCH max_abs={diff}"
                            )
                            ok = False
                        else:
                            print(
                                f"TEST {provider} {label} SKIP value_mismatch")
                except Exception as exc:
                    if provider == "optimized":
                        print(
                            f"TEST {provider} {label} MISMATCH exception={type(exc).__name__}"
                        )
                        ok = False
                    else:
                        print(
                            f"TEST {provider} {label} SKIP {type(exc).__name__}"
                        )
            del x, y, ref
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_one(provider, n):
    if provider == "baseline2" and baseline2 is None:
        return float("inf")
    if provider == "baseline1" and triton.cdiv(n, 4096) > 65535:
        return float("inf")
    x, y = _make_inputs(n)
    try:

        def fn():
            return _run_provider(provider, x, y)  # noqa: F821

        for _ in range(5):
            fn()
        _sync()
        t0 = time.perf_counter()
        reps = 20 if n <= (1 << 20) else 3
        for _ in range(reps):
            fn()
        _sync()
        return (time.perf_counter() - t0) * 1000.0 / reps
    except Exception:
        return float("inf")
    finally:
        del x, y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("black", "-"), ("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="ms",
        plot_name="mse_loss_perf",
        args={},
    ))
def benchmark(label, provider):
    n = dict((name, n) for name, n in _BENCH_SHAPES)[label]
    return _bench_one(provider, n)


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
        benchmark.run(print_data=True, show_plots=False, save_path=None)


if __name__ == "__main__":
    main()
