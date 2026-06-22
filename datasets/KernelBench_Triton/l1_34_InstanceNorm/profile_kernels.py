import argparse
import importlib.util
import pathlib
import sys
import time

import torch
import torch.nn as nn
import triton

ROOT = pathlib.Path(__file__).resolve().parent
INPUT_FILE = ROOT / "34_InstanceNorm.py"
BASE2_FILE = ROOT / "base_34_InstanceNorm.py"
OPT_FILE = ROOT / "opt_34_InstanceNorm.py"

_BENCH_SHAPES = [
    ("small_direct_8x64x32x32", 8, 64, 32, 32),
    ("medium_direct_16x64x128x128", 16, 64, 128, 128),
    ("large_direct_32x64x128x128", 32, 64, 128, 128),
]
_TEST_SHAPES = _BENCH_SHAPES + [
    ("irregular_direct_4x64x17x19", 4, 64, 17, 19),
]
_PROVIDER_FILES = {
    "baseline1": INPUT_FILE,
    "baseline2": BASE2_FILE,
    "optimized": OPT_FILE
}
_PROVIDER_NAMES = {
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton"
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


def _model(key, fresh=False):
    if fresh or key not in _MODEL_CACHE:
        if key == "torch_ref":
            m = nn.InstanceNorm2d(64,
                                  eps=1e-5,
                                  affine=False,
                                  track_running_stats=False)
        else:
            mod = _load(key, _PROVIDER_FILES[key])
            m = mod.ModelNew(*_init_args(mod))
        m = m.to(device="npu").eval()
        if fresh:
            return m
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


def _make_inputs(N, C, H, W):
    torch.manual_seed(0)
    return torch.rand((N, C, H, W), device="npu", dtype=torch.float32)


def _sync():
    torch.npu.synchronize()


def _bench(fn, warmup=3, rep=10):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(warmup):
            out = fn()
            _sync()
            del out
        t0 = time.perf_counter()
        for _ in range(rep):
            out = fn()
            _sync()
            del out
        return (time.perf_counter() - t0) * 1000.0 / rep


def _run(key, x, fresh=False):
    return _model("torch_ref" if key == "torch_ref" else key, fresh=fresh)(x)


def _unit_one(label, N, C, H, W):
    x = _make_inputs(N, C, H, W)
    ref_out = _run("torch_ref", x, fresh=True)
    _sync()
    ok_opt = True
    for key in _PROVIDER_NAMES:
        try:
            out = _run(key, x, fresh=True)
            _sync()
            same = torch.allclose(out, ref_out, rtol=1e-3, atol=1e-3)
            max_err = (out - ref_out).abs().max().item()
            print(
                f"TEST {key} eval {label} {'PASS' if same else 'MISMATCH'} max_err={max_err:.6g}"
            )
            if key == "optimized" and not same:
                ok_opt = False
            del out
        except Exception as e:
            tag = "FAIL" if key == "optimized" else "INFO"
            print(f"TEST {key} eval {label} {tag} {type(e).__name__}: {e}")
            if key == "optimized":
                ok_opt = False
    del ref_out, x
    return ok_opt


def unit_test():
    ok = True
    for shape in _TEST_SHAPES:
        ok = _unit_one(*shape) and ok
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
        plot_name="instancenorm2d-performance",
        args={},
    ))
def bench(label, provider):
    _, N, C, H, W = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = _make_inputs(N, C, H, W)
    try:
        return _bench(lambda: _run(provider, x, fresh=False))
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
