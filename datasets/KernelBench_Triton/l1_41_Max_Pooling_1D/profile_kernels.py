import argparse
import importlib.util
import math
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
import triton

ROOT = pathlib.Path(__file__).resolve().parent
_FILES = {
    "baseline1": ROOT / "41_Max_Pooling_1D.py",
    "baseline2": ROOT / "base_41_Max_Pooling_1D.py",
    "optimized": ROOT / "opt_41_Max_Pooling_1D.py",
}
_MODULES = {}
_MODELS = {}

DEFAULT_INIT = [8, 1, 4, 3, False]
_BENCH_SHAPES = [
    ("direct_small", 1, 2, 256),
    ("persistent_synthetic", 256, 256, 256),
    ("target_original", 64, 192, 65536),
]
_INDEX_TEST_SHAPES = [("index_direct", 1, 2, 256)]


def _load(key):
    if key in _MODULES:
        return _MODULES[key]
    path = _FILES[key]
    name = f"k_{key}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULES[key] = mod
    return mod


def _init_args(return_indices=False):
    args = list(DEFAULT_INIT)
    args[-1] = return_indices
    return args


def _model(key, return_indices=False):
    cache_key = (key, return_indices)
    if cache_key not in _MODELS:
        mod = _load(key)
        init = _init_args(return_indices)
        if init == [()]:
            init = []
        _MODELS[cache_key] = mod.ModelNew(*init).to(device="npu").eval()
    return _MODELS[cache_key]


def _make_input(N, C, L, dtype=torch.float32):
    torch.manual_seed(0)
    return torch.rand((N, C, L), device="npu", dtype=dtype)


def _run_torch_ref(x, return_indices=False):
    out = F.max_pool1d(x,
                       8,
                       stride=1,
                       padding=4,
                       dilation=3,
                       ceil_mode=False,
                       return_indices=return_indices)
    return out


def _run_provider(key, x, return_indices=False):
    return _model(key, return_indices=return_indices)(x)


def _as_tuple(y):
    return y if isinstance(y, tuple) else (y, )


def _compare(got, ref):
    got_t = _as_tuple(got)
    ref_t = _as_tuple(ref)
    if len(got_t) != len(ref_t):
        return False, "tuple_len"
    for i, (g, r) in enumerate(zip(got_t, ref_t)):
        if g.dtype.is_floating_point:
            diff = (
                g.to(torch.float32) -
                r.to(torch.float32)).abs().max().item() if g.numel() else 0.0
            ok = bool(diff <= 1e-3) or bool(
                torch.allclose(g, r, rtol=1e-3, atol=1e-3))
        else:
            ok = torch.equal(g, r)
            diff = 0.0
        if not bool(ok):
            if not g.dtype.is_floating_point:
                diff = (g.to(torch.float32) - r.to(torch.float32)
                        ).abs().max().item() if g.numel() else 0.0
            return False, f"out{i}_max_diff={diff}"
    return True, ""


def unit_test():
    providers = ["baseline1", "baseline2", "optimized"]
    all_opt_ok = True
    for label, N, C, L in _BENCH_SHAPES:
        x = _make_input(N, C, L)
        ref = _run_torch_ref(x, return_indices=False)
        torch.npu.synchronize()
        for key in providers:
            n_out = (L + 2 * DEFAULT_INIT[2] - DEFAULT_INIT[3] *
                     (DEFAULT_INIT[0] - 1) - 1) // DEFAULT_INIT[1] + 1
            n_blk = math.ceil(n_out / 128)
            if key in ("baseline1", "baseline2") and (N * C * n_blk > 65535
                                                      or N * C > 65535
                                                      or n_blk > 65535):
                print(f"TEST {key} {label} SKIP coreDim_guard")
                continue
            try:
                got = _run_provider(key, x, return_indices=False)
                torch.npu.synchronize()
                ok, msg = _compare(got, ref)
                if ok:
                    print(f"TEST {key} {label} PASS")
                else:
                    print(f"TEST {key} {label} MISMATCH {msg}")
                    if key == "optimized":
                        all_opt_ok = False
            except Exception as e:
                print(f"TEST {key} {label} SKIP {type(e).__name__}: {e}")
                if key == "optimized":
                    all_opt_ok = False
        del x, ref
        torch.npu.empty_cache()

    for label, N, C, L in _INDEX_TEST_SHAPES:
        x = _make_input(N, C, L)
        ref = _run_torch_ref(x, return_indices=False)
        try:
            got = _run_provider("optimized", x, return_indices=True)
            torch.npu.synchronize()
            got_values = got[0] if isinstance(got, tuple) else got
            ok, msg = _compare(got_values, ref)
            print(
                f"TEST optimized {label} {'PASS' if ok else 'MISMATCH ' + msg}"
            )
            all_opt_ok = all_opt_ok and ok
        except Exception as e:
            print(f"TEST optimized {label} SKIP {type(e).__name__}: {e}")
            all_opt_ok = False
        del x, ref
        torch.npu.empty_cache()

    print("UNIT_TEST PASS" if all_opt_ok else "UNIT_TEST_FAILED")
    return all_opt_ok


def _bench_once(fn, warmup=5, rep=20):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
            torch.npu.synchronize()
        times = []
        for _ in range(rep):
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return sum(times) / len(times)


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
        styles=[("black", "-"), ("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="ms",
        plot_name="maxpool1d-performance",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, N, C, L = shape
    n_out = (L + 2 * DEFAULT_INIT[2] - DEFAULT_INIT[3] *
             (DEFAULT_INIT[0] - 1) - 1) // DEFAULT_INIT[1] + 1
    n_blk = math.ceil(n_out / 128)
    if provider in ("baseline1", "baseline2") and (N * C * n_blk > 65535
                                                   or N * C > 65535
                                                   or n_blk > 65535):
        print(f"INFO {provider} {label} INF coreDim_guard")
        return float("inf")
    x = _make_input(N, C, L)
    if provider == "torch":

        def fn():
            return _run_torch_ref(x, return_indices=False)  # noqa: F821
    else:

        def fn():
            return _run_provider(
                provider,
                x,  # noqa: F821
                return_indices=False)

    try:
        ms = _bench_once(fn)
    except Exception as e:
        print(f"INFO {provider} {label} INF {type(e).__name__}: {e}")
        ms = float("inf")
    del x
    torch.npu.empty_cache()
    return ms


def run_bench():
    benchmark.run(print_data=True, show_plots=False, save_path=str(ROOT))


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
        run_bench()


if __name__ == "__main__":
    main()
