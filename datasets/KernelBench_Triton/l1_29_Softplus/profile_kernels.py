import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
_MAX_PROGRAMS = 65535
_BASELINE1_BLOCK = 4096
_OPT_BLOCK = 8192
_BENCH_SHAPES = [
    ("direct_irregular", (257, 333)),
    ("direct_medium", (1024, 1024)),
    ("persistent_original", (4096, 393216)),
]


def _load(stem: str):
    path = ROOT / stem
    name = "k_l1_29_" + stem.replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


MODS = {
    "baseline1": _load("29_Softplus.py"),
    "optimized": _load("opt_29_Softplus.py"),
}
if (ROOT / "base_29_Softplus.py").exists():
    MODS["baseline2"] = _load("base_29_Softplus.py")

_MODEL_CACHE = {}


def _model(key: str):
    if key not in _MODEL_CACHE:
        mod = MODS[key]
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        if init == [()]:
            init = []
        _MODEL_CACHE[key] = mod.ModelNew(*init).to(device="npu").eval()
    return _MODEL_CACHE[key]


def _make_inputs(shape):
    torch.manual_seed(0)
    return torch.rand(shape, device="npu", dtype=torch.float32)


def _run_torch_ref(x):
    return F.softplus(x, beta=1, threshold=20)


def _skip_reason(key: str, x) -> str | None:
    n = x.numel()
    if key == "baseline1" and triton.cdiv(n, _BASELINE1_BLOCK) > _MAX_PROGRAMS:
        return "coreDim_overflow_baseline1"
    return None


def _run_provider(key: str, x):
    if key == "torch":
        return _run_torch_ref(x)
    reason = _skip_reason(key, x)
    if reason:
        raise RuntimeError(reason)
    with torch.no_grad():
        return _model(key)(x)


def _sync():
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _bench_callable(fn):
    # triton.testing.do_bench is unreliable on torch_npu here; synchronize each
    # iteration so custom Triton launches cannot be under-timed by async queuing.
    for _ in range(3):
        out = fn()
        _sync()
        _ = float(out.reshape(-1)[0].item())
        del out
    t0 = time.perf_counter()
    for _ in range(10):
        out = fn()
        _sync()
        _ = float(out.reshape(-1)[0].item())
        del out
    return (time.perf_counter() - t0) * 1000.0 / 10.0


def _allclose(a, b):
    _sync()
    diff = (a - b).abs()
    max_abs = diff.max().item() if a.numel() else 0.0
    nonfinite = (~torch.isfinite(diff)).sum().item() if a.numel() else 0
    ok = (a.shape == b.shape) and (int(nonfinite) == 0) and (float(max_abs)
                                                             <= 1e-3)
    return bool(ok), float(max_abs)


def unit_test():
    all_opt_ok = True
    for label, shape in _BENCH_SHAPES:
        x = _make_inputs(shape)
        ref = _run_torch_ref(x)
        _sync()
        for key in ["baseline1", "baseline2", "optimized"]:
            if key not in MODS:
                continue
            provider_name = {
                "baseline1": "baseline1",
                "baseline2": "baseline2",
                "optimized": "optimized"
            }[key]
            reason = _skip_reason(key, x)
            if reason:
                print(f"TEST {label} {provider_name} SKIP {reason}")
                continue
            try:
                out = _run_provider(key, x)
                ok, max_abs = _allclose(out, ref)
                print(
                    f"TEST {label} {provider_name} {'PASS' if ok else 'MISMATCH'} max_abs={max_abs:.6g}"
                )
                if key == "optimized" and not ok:
                    all_opt_ok = False
            except Exception as exc:
                print(
                    f"TEST {label} {provider_name} INFO_EXCEPTION {type(exc).__name__}: {exc}"
                )
                if key == "optimized":
                    all_opt_ok = False
        del x, ref
        _sync()
    print("UNIT_TEST PASS" if all_opt_ok else "UNIT_TEST_FAILED")
    return all_opt_ok


_PROVIDERS = ["torch", "baseline1"] + (["baseline2"] if "baseline2" in MODS
                                       else []) + ["optimized"]
_LINE_NAMES = ["PyTorch / ACL", "Baseline Triton1"
               ] + (["Baseline Triton2"]
                    if "baseline2" in MODS else []) + ["Optimized Triton"]
_STYLES = [
    ("black", "-"), ("blue", "-")
] + ([("green", "-")] if "baseline2" in MODS else []) + [("red", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=_LINE_NAMES,
        styles=_STYLES,
        ylabel="ms",
        plot_name="softplus-l1-29",
        args={},
    ))
def bench(label, provider):
    shape = dict((name, shp) for name, shp in _BENCH_SHAPES)[label]
    x = _make_inputs(shape)
    reason = _skip_reason(provider, x) if provider != "torch" else None
    if reason:
        print(f"INFO benchmark {label} {provider} skipped: {reason}")
        return float("inf")
    try:

        def fn():
            return _run_provider(provider, x)

        ms = _bench_callable(fn)
        return ms
    except Exception as exc:
        print(
            f"INFO benchmark {label} {provider} exception: {type(exc).__name__}: {exc}"
        )
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
        bench.run(print_data=True,
                  show_plots=False,
                  save_path=str(ROOT / "remote_results"))


if __name__ == "__main__":
    main()
