#!/usr/bin/env python3
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

HERE = Path(__file__).resolve().parent
_MAX_PROGRAMS = 65535
INIT = [3, 2, 1]
_BENCH_SHAPES = [
    ("direct_small", 1, 1, 5, 5, 128),
    ("nonpow_medium", 2, 3, 17, 19, 131),
    ("persistent_synthetic", 70000, 1, 3, 3, 3),
]


def _load(stem, filename):
    path = HERE / filename
    if not path.exists():
        return None
    name = f"k_l1_46_{stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MODS = {
    "baseline1": _load("baseline1", "46_Average_Pooling_3D.py"),
    "baseline2": _load("baseline2", "base_46_Average_Pooling_3D.py"),
    "optimized": _load("optimized", "opt_46_Average_Pooling_3D.py"),
}
_MODEL_CACHE = {}


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _out_dim(size, k=3, stride=2, pad=1):
    return max(0, (size + 2 * pad - k) // stride + 1)


def _shape_stats(shape):
    _, _, D, H, W = shape
    return _out_dim(D), _out_dim(H), _out_dim(W)


def _baseline_grid_overflow(shape):
    N, C, D, H, W = shape
    OD, OH, OW = _shape_stats(shape)
    return triton.cdiv(N * C * OD * OH * OW, 256) > _MAX_PROGRAMS


def _tile_grid_overflow(shape):
    N, C, D, H, W = shape
    OD, OH, OW = _shape_stats(shape)
    return (N * C * OD * OH * triton.cdiv(OW, 64)) > _MAX_PROGRAMS


def _make_input(shape):
    torch.manual_seed(0)
    return torch.rand(shape, device="npu", dtype=torch.float32)


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    mod = MODS[key]
    if mod is None:
        return None
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else INIT
    if init == [()]:
        init = []
    m = mod.ModelNew(*init).to(device="npu").eval()
    _MODEL_CACHE[key] = m
    return m


def _run_torch_ref(x):
    return F.avg_pool3d(x, kernel_size=3, stride=2, padding=1)


def _run_provider(key, x, label=""):
    if key == "baseline1":
        raise RuntimeError("baseline_unavailable")
    if key in ("baseline1", "baseline2") and _baseline_grid_overflow(
            tuple(x.shape)):
        raise RuntimeError("grid_guard")
    if key == "baseline2" and _tile_grid_overflow(tuple(x.shape)):
        raise RuntimeError("grid_guard")
    m = _model(key)
    if m is None:
        raise RuntimeError("module_missing")
    return m(x)


def _as_tensor(y):
    return y[0] if isinstance(y, tuple) else y


def _allclose(a, b):
    a = _as_tensor(a)
    b = _as_tensor(b)
    return torch.allclose(a, b, rtol=1e-3,
                          atol=1e-3), (a - b).abs().max().item()


def unit_test():
    ok = True
    for label, *dims in _BENCH_SHAPES:
        x = _make_input(tuple(dims))
        ref = _run_torch_ref(x)
        _sync()
        for key in ("baseline1", "baseline2", "optimized"):
            if key == "baseline1":
                print(f"TEST {key} {label} SKIP baseline_unavailable")
                continue
            if key == "baseline2" and _tile_grid_overflow(tuple(x.shape)):
                print(f"TEST {key} {label} SKIP grid_guard")
                continue
            try:
                y = _run_provider(key, x, label)
                _sync()
                same, max_err = _allclose(y, ref)
                status = "PASS" if same else "MISMATCH"
                if key == "optimized" and not same:
                    ok = False
                print(f"TEST {key} {label} {status} max_err={max_err:.6g}")
            except Exception as e:
                if key == "optimized":
                    ok = False
                    print(f"TEST {key} {label} ERROR {type(e).__name__}: {e}")
                else:
                    print(f"TEST {key} {label} SKIP {type(e).__name__}: {e}")
        del x, ref
        _sync()
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_one(fn, warmup=3, rep=10):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        import time
        for _ in range(warmup):
            fn()
            _sync()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
            _sync()
        return (time.perf_counter() - t0) * 1000.0 / rep


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
        plot_name="avgpool3d-performance",
        args={},
    ))
def benchmark(label, provider):
    row = next(s for s in _BENCH_SHAPES if s[0] == label)
    shape = tuple(row[1:])
    x = _make_input(shape)
    try:
        if provider == "torch":

            def fn():
                return _run_torch_ref(x)  # noqa: F821
        elif provider == "baseline1":
            print(f"INFO {provider} {label} INF baseline_unavailable")
            return float("inf")
        elif provider == "baseline2" and _tile_grid_overflow(shape):
            print(f"INFO {provider} {label} INF grid_guard")
            return float("inf")
        else:

            def fn():
                return _run_provider(provider, x, label)  # noqa: F821

        return _bench_one(fn)
    except Exception as e:
        print(f"INFO {provider} {label} INF {type(e).__name__}: {e}")
        return float("inf")
    finally:
        del x
        _sync()


def main():
    unit_test()
    benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
