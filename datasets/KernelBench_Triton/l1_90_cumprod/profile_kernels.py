import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton

HERE = Path(__file__).resolve().parent


def _load(fname, key):
    path = HERE / fname
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, key):
    try:
        return _load(fname, key)
    except Exception as exc:
        print(f"TEST {key} import SKIP {type(exc).__name__}")
        return None


mods = {
    "baseline1": _load("90_cumprod.py", "baseline1"),
    "baseline2": _load_optional("base_90_cumprod.py", "baseline2"),
    "optimized": _load("opt_90_cumprod.py", "optimized"),
}
_MODEL_CACHE = {}

_BENCH_SHAPES = [
    ("small_2d", 128, 257),
    ("medium_2d", 1024, 1024),
    ("required_32768x32768", 32768, 32768),
]
_UNIT_ONLY_SHAPES = [
    ("persistent_rows", 70000, 8),
]
_PROVIDERS = ["baseline1", "baseline2", "optimized"]


def _init_args(mod, dim=1):
    if mod is None:
        return None
    if hasattr(mod, "get_init_inputs") and dim == 1:
        init = mod.get_init_inputs()
        if init == [()]:
            init = []
        return list(init)
    return [dim]


def _model(key, dim=1):
    mod = mods[key]
    init = _init_args(mod, dim)
    cache_key = (key, tuple(init) if init is not None else None)
    if mod is None:
        return None
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = mod.ModelNew(*init).to(device="npu").eval()
    return _MODEL_CACHE[cache_key]


def _make_input(M, N):
    torch.manual_seed(0)
    # Match source get_inputs(): torch.rand(batch_size, *input_shape), fp32, contiguous.
    return torch.rand((M, N), device="npu", dtype=torch.float32)


def _run_torch_ref(x, dim=1):
    return torch.cumprod(x, dim=dim)


def _baseline1_grid_guard(M, N):
    # baseline launches one program per row; guard Ascend coreDim before it can poison context.
    return M <= 65535


def _run_provider(key, x, dim=1):
    if key == "baseline2" and mods[key] is None:
        raise RuntimeError("baseline2_unavailable")
    if key == "baseline1" and not _baseline1_grid_guard(
            x.shape[0], x.shape[1]):
        raise RuntimeError("grid_guard")
    model = _model(key, dim)
    return model(x)


def _check_one(label, M, N, key, dim=1):
    x = _make_input(M, N)
    ref = _run_torch_ref(x, dim)
    try:
        out = _run_provider(key, x, dim)
        torch.npu.synchronize()
        ok = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
        if ok:
            print(f"TEST {key} {label} PASS")
            return True
        max_err = (out - ref).abs().max().item()
        print(
            f"TEST {key} {label} SKIP value_difference max_abs={max_err:.6g}")
        return key != "optimized"
    except Exception as exc:
        if key == "optimized":
            print(f"TEST {key} {label} MISMATCH {type(exc).__name__}")
            return False
        print(f"TEST {key} {label} SKIP {type(exc).__name__}")
        return True


def unit_test():
    all_ok = True
    for label, M, N in _BENCH_SHAPES:
        for key in _PROVIDERS:
            all_ok = _check_one(label, M, N, key) and all_ok
    for label, M, N in _UNIT_ONLY_SHAPES:
        all_ok = _check_one(label, M, N, "optimized") and all_ok
    print("UNIT_TEST PASS" if all_ok else "UNIT_TEST_FAILED")
    return all_ok


def _bench_provider(label, M, N, provider):
    if provider == "PyTorch / ACL":
        fn_key = "torch"
    elif provider == "Baseline Triton1":
        fn_key = "baseline1"
    elif provider == "Baseline Triton2":
        fn_key = "baseline2"
    else:
        fn_key = "optimized"

    x = _make_input(M, N)
    if fn_key == "baseline1" and not _baseline1_grid_guard(M, N):
        return float("inf")
    if fn_key == "baseline2" and mods["baseline2"] is None:
        return float("inf")

    def fn():
        if fn_key == "torch":
            return _run_torch_ref(x)
        return _run_provider(fn_key, x)

    try:
        warmup = 3 if M * N >= 1024 * 1024 else 10
        rep = 10 if M * N >= 1024 * 1024 else 50
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception as exc:
        print(f"BENCH {fn_key} {label} INF {type(exc).__name__}")
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("black", "-"), ("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="cumprod-performance",
        args={},
    ))
def benchmark(label, provider):
    shape = {s[0]: s for s in _BENCH_SHAPES}[label]
    _, M, N = shape
    return _bench_provider(label, M, N, provider)


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
