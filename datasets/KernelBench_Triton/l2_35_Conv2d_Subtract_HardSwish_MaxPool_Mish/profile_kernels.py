import argparse
import importlib.util
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
import triton

HERE = pathlib.Path(__file__).resolve().parent
INPUT_FILE = "35_Conv2d_Subtract_HardSwish_MaxPool_Mish.py"
BASE_FILE = "base_35_Conv2d_Subtract_HardSwish_MaxPool_Mish.py"
OPT_FILE = "opt_35_Conv2d_Subtract_HardSwish_MaxPool_Mish.py"


def _load(fname, name):
    path = HERE / fname
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mods = {}
load_errors = {}
for key, fname in [("baseline1", INPUT_FILE), ("baseline2", BASE_FILE),
                   ("optimized", OPT_FILE)]:
    try:
        mods[key] = _load(fname, f"k_{key}_{fname.replace('.', '_')}")
    except Exception as exc:
        load_errors[key] = type(exc).__name__
        mods[key] = None

_BENCH_SHAPES = [
    # label, batch, in_channels, out_channels, height, width, kernel_size, subtract, pool_k
    ("default_k2", 128, 64, 128, 128, 128, 3, 0.5, 2),
    ("small_k2", 4, 64, 128, 65, 67, 3, 0.5, 2),
    ("fallback_k3", 2, 64, 128, 33, 35, 3, 0.5, 3),
]

_MODEL_CACHE = {}


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _device():
    return torch.device("npu")


def _make_input(shape):
    label, B, IC, OC, H, W, K, subtract, pool_k = shape
    torch.manual_seed(123)
    return torch.rand(B, IC, H, W, device=_device(), dtype=torch.float32)


def _init_args(shape):
    label, B, IC, OC, H, W, K, subtract, pool_k = shape
    return [IC, OC, K, subtract, pool_k]


def _model(key, shape):
    cache_key = (key, shape[0])
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    mod = mods.get(key)
    if mod is None:
        raise RuntimeError(
            f"module_unavailable_{key}_{load_errors.get(key, 'unknown')}")
    torch.manual_seed(0)
    model = mod.ModelNew(*_init_args(shape)).to(_device()).eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _torch_ref_model(shape):
    torch.manual_seed(0)
    label, B, IC, OC, H, W, K, subtract, pool_k = shape
    m = torch.nn.Conv2d(IC, OC, K).to(_device()).eval()
    return m


_REF_CACHE = {}


def _run_torch_ref(x, shape):
    if shape[0] not in _REF_CACHE:
        _REF_CACHE[shape[0]] = _torch_ref_model(shape)
    conv = _REF_CACHE[shape[0]](x)
    subtract = shape[7]
    pool_k = shape[8]
    v = conv - subtract
    hs = v * torch.clamp(v + 3.0, min=0.0, max=6.0) / 6.0
    pooled = F.max_pool2d(hs, pool_k)
    return pooled * torch.tanh(F.softplus(pooled))


def _run_provider(key, x, shape):
    if key == "torch":
        return _run_torch_ref(x, shape)
    return _model(key, shape)(x)


def _assert_close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3, equal_nan=True)


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_input(shape)
        ref = _run_torch_ref(x, shape)
        _sync()
        for key, pretty in [("baseline1", "Baseline Triton1"),
                            ("baseline2", "Baseline Triton2"),
                            ("optimized", "Optimized Triton")]:
            try:
                out = _run_provider(key, x, shape)
                _sync()
                _assert_close(out, ref)
                print(f"CHECK {pretty} {label} PASS")
            except Exception as exc:
                if key == "optimized":
                    ok = False
                    print(f"CHECK {pretty} {label} FAIL {type(exc).__name__}")
                else:
                    print(
                        f"INFO {pretty} {label} unavailable_or_mismatch {type(exc).__name__}"
                    )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _manual_bench(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


def _bench_one(provider, label):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = _make_input(shape)
    try:
        _run_provider(provider, x, shape)
        _sync()
    except Exception as exc:
        print(f"INFO bench_preskip {provider} {label} {type(exc).__name__}")
        return float("inf")
    try:
        return triton.testing.do_bench(
            lambda: _run_provider(provider, x, shape),
            warmup=10,
            rep=30,
            return_mode="mean")
    except Exception:
        return _manual_bench(lambda: _run_provider(provider, x, shape),
                             warmup=5,
                             rep=10)


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
        plot_name="conv2d_subtract_hswish_maxpool_mish",
        args={},
    ))
def bench(label, provider):
    return _bench_one(provider, label)


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
