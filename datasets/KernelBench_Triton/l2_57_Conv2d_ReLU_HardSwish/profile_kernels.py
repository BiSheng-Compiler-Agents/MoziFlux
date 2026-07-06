import argparse
import importlib.util
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
import triton

HERE = pathlib.Path(__file__).resolve().parent
INPUT_FILE = "57_Conv2d_ReLU_HardSwish.py"
OPT_FILE = "opt_57_Conv2d_ReLU_HardSwish.py"


def _load(fname, name):
    path = HERE / fname
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mods = {}
load_errors = {}
for key, fname in [("baseline1", INPUT_FILE), ("optimized", OPT_FILE)]:
    try:
        mods[key] = _load(fname, f"k_{key}_{fname.replace('.', '_')}")
    except Exception as exc:
        load_errors[key] = type(exc).__name__
        mods[key] = None

# Reference base_*.py is intentionally not read in this sandbox; keep parser-visible column.
mods["baseline2"] = None
load_errors["baseline2"] = "SandboxReferenceReadForbidden"

_BENCH_SHAPES = [
    ("default", 128, 8, 64, 128, 128, 3),
    ("small_irregular", 4, 8, 64, 65, 67, 3),
    ("batch1", 1, 8, 64, 33, 35, 3),
]

_MODEL_CACHE = {}
_REF_CACHE = {}


def _device():
    return torch.device("npu")


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _make_input(shape):
    label, B, IC, OC, H, W, K = shape
    torch.manual_seed(123)
    return torch.rand(B, IC, H, W, device=_device(), dtype=torch.float32)


def _init_args(shape):
    label, B, IC, OC, H, W, K = shape
    return [IC, OC, K]


def _model(key, shape):
    cache_key = (key, shape[0])
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    mod = mods.get(key)
    if mod is None:
        raise RuntimeError(f"module_unavailable_{key}_{load_errors.get(key, 'unknown')}")
    torch.manual_seed(0)
    model = mod.ModelNew(*_init_args(shape)).to(_device()).eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _torch_ref_model(shape):
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(shape[2], shape[3], shape[6]).to(_device()).eval()
    return conv


def _run_torch_ref(x, shape):
    if shape[0] not in _REF_CACHE:
        _REF_CACHE[shape[0]] = _torch_ref_model(shape)
    y = _REF_CACHE[shape[0]](x)
    r = torch.relu(y)
    return r * torch.clamp(r + 3.0, min=0.0, max=6.0) * (1.0 / 6.0)


def _run_provider(key, x, shape):
    if key == "torch":
        return _run_torch_ref(x, shape)
    return _model(key, shape)(x)


def _max_abs(a, b):
    return (a - b).abs().max().detach().float().cpu().item()


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_input(shape)
        ref = _run_torch_ref(x, shape)
        _sync()
        for key, pretty in [("baseline1", "Baseline Triton1"), ("baseline2", "Baseline Triton2"), ("optimized", "Optimized Triton")]:
            if key == "baseline2":
                print(f"TEST {pretty} {label}: SKIP reference_read_forbidden max_abs=inf")
                continue
            try:
                out = _run_provider(key, x, shape)
                _sync()
                diff = _max_abs(out, ref)
                if diff <= 1e-3:
                    print(f"TEST {pretty} {label}: PASS max_abs={diff:.6g}")
                else:
                    if key == "optimized":
                        ok = False
                    print(f"TEST {pretty} {label}: MISMATCH max_abs={diff:.6g}")
            except Exception as exc:
                if key == "optimized":
                    ok = False
                    print(f"TEST {pretty} {label}: FAIL {type(exc).__name__} max_abs=inf")
                else:
                    print(f"TEST {pretty} {label}: SKIP {type(exc).__name__} max_abs=inf")

    # Forced persistent path without allocating a grid-cap-sized tensor.
    try:
        opt = mods["optimized"]
        old = opt._MAX_PROGRAMS
        opt._MAX_PROGRAMS = 1
        shape = _BENCH_SHAPES[0]
        _MODEL_CACHE.pop(("optimized", shape[0]), None)
        x = _make_input(shape)
        ref = _run_torch_ref(x, shape)
        out = _run_provider("optimized", x, shape)
        _sync()
        diff = _max_abs(out, ref)
        if diff <= 1e-3:
            print(f"TEST Optimized Triton forced_persistent: PASS max_abs={diff:.6g}")
        else:
            ok = False
            print(f"TEST Optimized Triton forced_persistent: MISMATCH max_abs={diff:.6g}")
    except Exception as exc:
        ok = False
        print(f"TEST Optimized Triton forced_persistent: FAIL {type(exc).__name__} max_abs=inf")
    finally:
        try:
            opt._MAX_PROGRAMS = old
            _MODEL_CACHE.pop(("optimized", _BENCH_SHAPES[0][0]), None)
        except Exception:
            pass

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
    if provider == "baseline2":
        print(f"INFO bench_preskip baseline2 {label} reference_read_forbidden")
        return float("inf")
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = _make_input(shape)
    try:
        _run_provider(provider, x, shape)
        _sync()
    except Exception as exc:
        print(f"INFO bench_preskip {provider} {label} {type(exc).__name__}")
        return float("inf")
    try:
        return triton.testing.do_bench(lambda: _run_provider(provider, x, shape), warmup=10, rep=30, return_mode="mean")
    except Exception:
        return _manual_bench(lambda: _run_provider(provider, x, shape), warmup=5, rep=10)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "baseline1", "baseline2", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("black", "-"), ("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="ms",
        plot_name="conv2d_relu_hardswish",
        args={},
    )
)
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
