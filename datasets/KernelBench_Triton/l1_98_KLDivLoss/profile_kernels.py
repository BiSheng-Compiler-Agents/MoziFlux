import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
        print(f"INFO {key} unavailable {type(exc).__name__}")
        return None


_MODS = {
    "baseline1": _load("98_KLDivLoss.py", "baseline1"),
    "baseline2": _load_optional("base_98_KLDivLoss.py", "baseline2"),
    "optimized": _load("opt_98_KLDivLoss.py", "optimized"),
}
_MODEL_CACHE = {}

_BENCH_SHAPES = [
    ("small_8x256", 8, 256, "random"),
    ("medium_128x4096", 128, 4096, "random"),
    ("irregular_17x3000", 17, 3000, "random"),
    ("exact_16384x16384", 16384, 16384, "fill"),
]

_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_PROVIDER_NAMES = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _model(key, init_args=None):
    if init_args is None:
        init_args = []
    if init_args == [()]:
        init_args = []
    cache_key = (key, tuple(init_args))
    if cache_key not in _MODEL_CACHE:
        mod = _MODS[key]
        if mod is None:
            return None
        _MODEL_CACHE[cache_key] = mod.ModelNew(*init_args).to(
            device="npu").eval()
    return _MODEL_CACHE[cache_key]


def _make_inputs(B, D, mode):
    if mode == "fill":
        p = torch.empty((B, D), device="npu",
                        dtype=torch.float32).fill_(0.9 / D)
        t = torch.empty((B, D), device="npu",
                        dtype=torch.float32).fill_(1.1 / D)
        return p, t
    torch.manual_seed(0)
    p = torch.rand((B, D), device="npu", dtype=torch.float32).softmax(dim=-1)
    t = torch.rand((B, D), device="npu", dtype=torch.float32).softmax(dim=-1)
    return p, t


def _run_torch_ref(p, t):
    return F.kl_div(p.log(), t, reduction="batchmean", log_target=False)


def _run_provider(provider, p, t):
    if provider == "torch":
        return _run_torch_ref(p, t)
    mod = _MODS.get(provider)
    if mod is None:
        raise RuntimeError("provider_unavailable")
    model = _model(provider)
    return model(p, t)


def _allclose_scalar(a, b, rtol=1e-3, atol=1e-3):
    av = float(a.detach().float().cpu())
    bv = float(b.detach().float().cpu())
    diff = abs(av - bv)
    tol = atol + rtol * abs(bv)
    return diff <= tol, av, bv, diff, tol


def unit_test():
    ok = True
    for label, B, D, mode in _BENCH_SHAPES:
        p, t = _make_inputs(B, D, mode)
        ref = _run_torch_ref(p, t)
        _sync()
        for provider in ["baseline1", "baseline2", "optimized"]:
            name = provider
            if _MODS.get(provider) is None:
                print(f"TEST {name} {label} SKIP provider_unavailable")
                continue
            try:
                out = _run_provider(provider, p, t)
                _sync()
                passed, got, exp, diff, tol = _allclose_scalar(out, ref)
                if passed:
                    print(
                        f"TEST {name} {label} PASS got={got:.6g} ref={exp:.6g}"
                    )
                else:
                    if provider == "optimized":
                        print(
                            f"TEST {name} {label} MISMATCH got={got:.6g} ref={exp:.6g} diff={diff:.3g} tol={tol:.3g}"
                        )
                        ok = False
                    else:
                        print(
                            f"TEST {name} {label} SKIP value_diff diff={diff:.3g} tol={tol:.3g}"
                        )
            except Exception as exc:
                if provider == "optimized":
                    print(
                        f"TEST {name} {label} MISMATCH runtime {type(exc).__name__}"
                    )
                    ok = False
                else:
                    print(
                        f"TEST {name} {label} SKIP runtime_{type(exc).__name__}"
                    )
        # Cover optimized diagnostic Triton fallback dispatch separately on bounded shapes.
        if B * D <= 1_000_000:
            try:
                fallback = _MODS["optimized"].ModelNew(True).to(
                    device="npu").eval()(p, t)
                _sync()
                passed, got, exp, diff, tol = _allclose_scalar(fallback, ref)
                if passed:
                    print(
                        f"TEST optimized_fallback {label} PASS got={got:.6g} ref={exp:.6g}"
                    )
                else:
                    print(
                        f"TEST optimized_fallback {label} MISMATCH got={got:.6g} ref={exp:.6g} diff={diff:.3g} tol={tol:.3g}"
                    )
                    ok = False
            except Exception as exc:
                print(
                    f"TEST optimized_fallback {label} MISMATCH runtime {type(exc).__name__}"
                )
                ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _manual_bench(fn, warmup, rep):
    for _ in range(warmup):
        fn()
        _sync()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
        _sync()
    return (time.perf_counter() - start) * 1000.0 / rep


def _bench_one(provider, label, B, D, mode):
    if provider != "torch" and _MODS.get(provider) is None:
        print(f"INFO {provider} {label} bench_inf provider_unavailable")
        return float("inf")
    p, t = _make_inputs(B, D, mode)

    def fn():
        return _run_provider(provider, p, t)

    warmup, rep = (3, 10) if B * D >= 50_000_000 else (10, 50)
    try:
        if hasattr(triton, "testing") and hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn,
                                           warmup=warmup,
                                           rep=rep,
                                           return_mode="mean")
        return _manual_bench(fn, warmup=warmup, rep=rep)
    except Exception as exc:
        print(
            f"INFO {provider} {label} bench_inf runtime_{type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[_PROVIDER_NAMES[p] for p in _PROVIDERS],
        styles=[("black", "-"), ("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="kl_div_loss_perf",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, B, D, mode = shape
    return _bench_one(provider, label, B, D, mode)


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
        bench.run(print_data=True, show_plots=False, save_path=None)


if __name__ == "__main__":
    main()
