import argparse
import importlib.util
import pathlib
import sys
import time

import torch
import torch.nn as nn
import triton

ROOT = pathlib.Path(__file__).resolve().parent
INPUT_FILE = ROOT / "33_BatchNorm.py"
BASE2_FILE = ROOT / "base_33_BatchNorm.py"
OPT_FILE = ROOT / "opt_33_BatchNorm.py"

_BENCH_SHAPES = [
    ("small_8x64x32x32", 8, 64, 32, 32),
    ("medium_16x64x128x128", 16, 64, 128, 128),
    ("target_64x64x512x512", 64, 64, 512, 512),
]
_PROVIDER_FILES = {
    "baseline1": INPUT_FILE,
    "baseline2": BASE2_FILE,
    "optimized": OPT_FILE,
}
_PROVIDER_NAMES = {
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}
_MODEL_CACHE = {}
_MOD_CACHE = {}


def _load(key, path):
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MOD_CACHE[key] = mod
    return mod


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    return init


def _model(key, train=True, fresh=False):
    cache_key = (key, train)
    if fresh or cache_key not in _MODEL_CACHE:
        if key == "torch_ref":
            m = nn.BatchNorm2d(64)
        else:
            mod = _load(key, _PROVIDER_FILES[key])
            m = mod.ModelNew(*_init_args(mod))
        m = m.to(device="npu").eval()
        if train:
            m.train()
        if fresh:
            return m
        _MODEL_CACHE[cache_key] = m
    return _MODEL_CACHE[cache_key]


def _make_inputs(N, C, H, W):
    torch.manual_seed(0)
    return torch.rand((N, C, H, W), device="npu", dtype=torch.float32)


def _run_torch_ref(x, train=True, fresh=False):
    return _model("torch_ref", train=train, fresh=fresh)(x)


def _run_provider(key, x, train=True, fresh=False):
    return _model(key, train=train, fresh=fresh)(x)


def _sync():
    torch.npu.synchronize()


def _bench(fn, warmup=10, rep=50):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
            _sync()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
            _sync()
        return (time.perf_counter() - t0) * 1000.0 / rep


def _copy_bn_state(src, dst):
    dst.weight.data.copy_(src.weight.data)
    dst.bias.data.copy_(src.bias.data)
    if src.running_mean is not None and dst.running_mean is not None:
        dst.running_mean.data.copy_(src.running_mean.data)
        dst.running_var.data.copy_(src.running_var.data)
        dst.num_batches_tracked.data.copy_(src.num_batches_tracked.data)


def _baseline_grid_overflows(key, N, C, H, W):
    # The input/golden row-wise kernels launch (C, N*H); guard invalid Ascend coreDim.
    return key in ("baseline1", "baseline2") and (C * N * H > 65535)


def _unit_one(label, N, C, H, W, train=True):
    x = _make_inputs(N, C, H, W)
    ref = _model("torch_ref", train=train, fresh=True)
    ref_out = ref(x)
    _sync()
    ok_opt = True
    mode = "train" if train else "eval"
    for key, name in _PROVIDER_NAMES.items():
        if _baseline_grid_overflows(key, N, C, H, W):
            print(f"TEST {key} {mode} {label} SKIP coreDim_guard")
            continue
        try:
            m = _model(key, train=train, fresh=True)
            if hasattr(m, "bn"):
                _copy_bn_state(ref, m.bn)
            out = m(x)
            _sync()
            same = torch.allclose(out, ref_out, rtol=1e-3, atol=1e-3)
            max_err = (out - ref_out).abs().max().item()
            status = "PASS" if same else "MISMATCH"
            print(f"TEST {key} {mode} {label} {status} max_err={max_err:.6g}")
            if key == "optimized" and not same:
                ok_opt = False
        except Exception as e:
            tag = "FAIL" if key == "optimized" else "INFO"
            print(f"TEST {key} {mode} {label} {tag} {type(e).__name__}: {e}")
            if key == "optimized":
                ok_opt = False
    return ok_opt


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        ok = _unit_one(*shape, train=True) and ok
    ok = _unit_one("eval_dispatch_4x64x17x19", 4, 64, 17, 19,
                   train=False) and ok
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("black", "-"), ("blue", "-"), ("green", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="batchnorm2d-performance",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, N, C, H, W = shape
    x = _make_inputs(N, C, H, W)
    try:
        if _baseline_grid_overflows(provider, N, C, H, W):
            print(f"INFO BENCH {provider} {label} inf coreDim_guard")
            return float("inf")
        if provider == "torch_ref":

            def fn():
                return _run_torch_ref(x, train=True, fresh=False)
        else:

            def fn():
                return _run_provider(provider, x, train=True, fresh=False)

        return _bench(fn)
    except Exception as e:
        print(f"INFO BENCH {provider} {label} inf {type(e).__name__}: {e}")
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
