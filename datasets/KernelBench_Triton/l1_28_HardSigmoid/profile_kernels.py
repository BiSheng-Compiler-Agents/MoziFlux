import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

HERE = Path(__file__).resolve().parent
INPUT_FILE = "28_HardSigmoid.py"
BASE_FILE = "base_28_HardSigmoid.py"
OPT_FILE = "opt_28_HardSigmoid.py"
OPT_BLOCK_SIZE = 8192
MAX_PROGRAMS = 65535
_BENCH_SHAPES = [
    ("direct_1M", (1024, 1024)),
    ("direct_irregular", (257, 4097)),
    ("persistent_original", (4096, 393216)),
]
_MODEL_CACHE = {}
_MOD_CACHE = {}


def _load(key, filename):
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    path = HERE / filename
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


def _model(key):
    files = {
        "baseline1": INPUT_FILE,
        "baseline2": BASE_FILE,
        "optimized": OPT_FILE
    }
    if key not in _MODEL_CACHE:
        mod = _load(key, files[key])
        _MODEL_CACHE[key] = mod.ModelNew(*_init_args(mod)).to(
            device="npu").eval()
    return _MODEL_CACHE[key]


def _make_inputs(shape):
    torch.manual_seed(0)
    return torch.rand(*shape, device="npu", dtype=torch.float32)


def _run_torch_ref(x):
    return F.hardsigmoid(x)


def _provider_grid_overflows(key, x):
    n = x.numel()
    if key == "baseline1":
        return triton.cdiv(n, 2048) > MAX_PROGRAMS
    return False


def _run_provider(key, x):
    if key == "baseline2" and not (HERE / BASE_FILE).exists():
        raise RuntimeError("base_28_HardSigmoid.py not present")
    if _provider_grid_overflows(key, x):
        raise RuntimeError("SKIP coreDim would exceed 65535 for this provider")
    with torch.no_grad():
        return _model(key)(x)


def _sync():
    torch.npu.synchronize()


def _bench_ms(fn, warmup=25, rep=200):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(min(5, warmup)):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        _sync()
        return (time.perf_counter() - t0) * 1000.0 / rep


def unit_test():
    ok_opt = True
    providers = [("baseline1", "Baseline Triton1"),
                 ("baseline2", "Baseline Triton2"),
                 ("optimized", "Optimized Triton")]
    for label, shape in _BENCH_SHAPES:
        x = _make_inputs(shape)
        ref = _run_torch_ref(x)
        _sync()
        for key, name in providers:
            try:
                out = _run_provider(key, x)
                _sync()
                torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
                print(f"TEST {label} {key} PASS")
            except Exception as e:
                msg = str(e).splitlines()[0]
                if key == "optimized":
                    ok_opt = False
                    print(f"TEST {label} {key} MISMATCH {msg}")
                else:
                    print(f"TEST {label} {key} INFO {msg}")
        del x, ref
        _sync()
    print("UNIT_TEST PASS" if ok_opt else "UNIT_TEST_FAILED")
    return ok_opt


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
        styles=[("blue", "-"), ("green", "-"), ("black", "--"), ("red", "-")],
        ylabel="Latency (ms)",
        plot_name="hardsigmoid-performance",
        args={},
    ))
def bench(label, provider):
    shape = dict((name, shp) for name, shp in _BENCH_SHAPES)[label]
    x = _make_inputs(shape)
    if provider == "torch":

        def fn():
            return _run_torch_ref(x)
    else:
        try:
            _run_provider(provider, x)
            _sync()
        except Exception as e:
            print(f"INFO bench {label} {provider}: {str(e).splitlines()[0]}")
            return float("inf")

        def fn():
            return _run_provider(provider, x)

    try:
        return _bench_ms(fn)
    except Exception as e:
        print(f"INFO bench {label} {provider}: {str(e).splitlines()[0]}")
        return float("inf")


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
        bench.run(print_data=True, show_plots=False, save_path=str(HERE))


if __name__ == "__main__":
    main()
