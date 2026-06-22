import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
FILES = {
    "baseline1": ROOT / "52_Argmin_over_a_dimension.py",
    "baseline2": ROOT / "base_52_Argmin_over_a_dimension.py",
    "optimized": ROOT / "opt_52_Argmin_over_a_dimension.py",
}
LINE_NAMES = [
    "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
]
LINE_VALS = ["torch", "baseline1", "baseline2", "optimized"]

_BENCH_SHAPES = [
    ("small_dim1", 4, 129, 127, 1),
    ("irregular_dim1", 8, 257, 255, 1),
    ("dim0_path", 16, 64, 65, 0),
    ("dim2_path", 8, 128, 129, 2),
    ("exact_target", 128, 4096, 4095, 1),
]
_TEST_ONLY_SHAPES = [
    ("two_d_fallback", (129, 127), 0),
]

_MODULES = {}
_MODELS = {}


def _load(key):
    if key in _MODULES:
        return _MODULES[key]
    path = FILES[key]
    name = f"k_{key}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _MODULES[key] = mod
    return mod


def _model(key, dim):
    cache_key = (key, int(dim))
    if cache_key not in _MODELS:
        mod = _load(key)
        _MODELS[cache_key] = mod.ModelNew(int(dim)).to(device="npu").eval()
    return _MODELS[cache_key]


def _make_3d_inputs(B, M, N, dtype=torch.float32):
    torch.manual_seed(0)
    return torch.rand((B, M, N), device="npu", dtype=dtype)


def _make_nd_inputs(shape, dtype=torch.float32):
    torch.manual_seed(0)
    return torch.rand(shape, device="npu", dtype=dtype)


def _run_torch_ref(x, dim):
    return torch.argmin(x, dim=int(dim))


def _run_provider(key, x, dim):
    return _model(key, dim)(x)


def _comparison_grid_overflows(provider, x, dim):
    if provider in ("torch", "optimized"):
        return False
    # The editable/read-only baselines launch one program per output element on the
    # movedim-last representation; exact_target dim=1 exceeds Ascend coreDim.
    out_elems = x.numel() // x.shape[int(dim)]
    return out_elems > 65535


def _check_one(provider, label, x, dim, ref):
    if _comparison_grid_overflows(provider, x, dim):
        print(f"TEST {provider} {label} SKIP grid_guard")
        return True
    if provider == "torch":
        got = _run_torch_ref(x, dim)
    else:
        got = _run_provider(provider, x, dim)
    torch.npu.synchronize()
    if got.shape != ref.shape or got.dtype != ref.dtype or not torch.equal(
            got.cpu(), ref.cpu()):
        if provider == "optimized":
            detail = ""
            try:
                bad = (got.cpu() != ref.cpu()).nonzero()
                if bad.numel() > 0:
                    i = tuple(int(v) for v in bad[0].tolist())
                    detail = f" first_bad={i} got={int(got.cpu()[i])} ref={int(ref.cpu()[i])}"
            except BaseException:
                detail = ""
            print(
                f"TEST optimized {label} MISMATCH shape={tuple(got.shape)} ref_shape={tuple(ref.shape)}{detail}"
            )
        else:
            print(f"TEST {provider} {label} SKIP comparison_mismatch")
        return provider != "optimized"
    pname = "optimized" if provider == "optimized" else provider
    print(f"TEST {pname} {label} PASS")
    return True


def unit_test():
    ok = True
    for label, B, M, N, dim in _BENCH_SHAPES:
        x = _make_3d_inputs(B, M, N)
        ref = _run_torch_ref(x, dim)
        torch.npu.synchronize()
        for provider in ["optimized", "baseline1", "baseline2"]:
            try:
                if not _check_one(provider, label, x, dim, ref):
                    ok = False
            except BaseException:
                if provider == "optimized":
                    print(
                        f"TEST optimized {label} MISMATCH provider_unavailable"
                    )
                    ok = False
                else:
                    print(f"TEST {provider} {label} SKIP provider_unavailable")
    for label, shape, dim in _TEST_ONLY_SHAPES:
        x = _make_nd_inputs(shape)
        ref = _run_torch_ref(x, dim)
        torch.npu.synchronize()
        try:
            if not _check_one("optimized", label, x, dim, ref):
                ok = False
        except BaseException:
            print(f"TEST optimized {label} MISMATCH provider_unavailable")
            ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_provider(provider, x, dim):
    if _comparison_grid_overflows(provider, x, dim):
        return float("inf")
    if provider == "torch":

        def fn():
            return _run_torch_ref(x, dim)
    else:

        def fn():
            return _run_provider(provider, x, dim)

    try:
        fn()
        torch.npu.synchronize()
        return triton.testing.do_bench(fn,
                                       warmup=5,
                                       rep=20,
                                       return_mode="mean")
    except BaseException:
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=[("black", "-"), ("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="argmin_over_dimension_perf",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, B, M, N, dim = shape
    x = _make_3d_inputs(B, M, N)
    return _bench_provider(provider, x, dim)


def _print_summary():
    vals = {}
    for label, B, M, N, dim in _BENCH_SHAPES:
        x = _make_3d_inputs(B, M, N)
        vals[label] = {}
        for p in LINE_VALS:
            vals[label][p] = _bench_provider(p, x, dim)
    print(
        "\nSummary ratios (Baseline Triton1 / Optimized, PyTorch / Optimized):"
    )
    ratios_b = []
    ratios_t = []
    for label, row in vals.items():
        opt = row["optimized"]
        b1 = row["baseline1"]
        th = row["torch"]
        rb = b1 / opt if math.isfinite(b1) and math.isfinite(
            opt) and opt > 0 else float("nan")
        rt = th / opt if math.isfinite(th) and math.isfinite(
            opt) and opt > 0 else float("nan")
        print(f"{label}: baseline1/opt={rb:.3f}, torch/opt={rt:.3f}")
        if math.isfinite(rb) and rb > 0:
            ratios_b.append(rb)
        if math.isfinite(rt) and rt > 0:
            ratios_t.append(rt)
    if ratios_b:
        print(
            f"GEOMEAN baseline1/optimized {math.exp(sum(math.log(v) for v in ratios_b)/len(ratios_b)):.3f}"
        )
    if ratios_t:
        print(
            f"GEOMEAN torch/optimized {math.exp(sum(math.log(v) for v in ratios_t)/len(ratios_t)):.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not args.test and not args.bench:
        args.test = True
        args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        benchmark.run(print_data=True, show_plots=False)
        _print_summary()


if __name__ == "__main__":
    main()
