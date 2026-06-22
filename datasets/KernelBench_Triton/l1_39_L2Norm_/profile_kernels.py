import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.testing

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "39_L2Norm_.py"
BASE_FILE = ROOT / "base_39_L2Norm_.py"
OPT_FILE = ROOT / "opt_39_L2Norm_.py"

_BENCH_SHAPES = [
    ("small_singlepass", (128, 1024)),
    ("irregular_medium", (513, 10000)),
    ("overflow_rows", (70000, 16)),
    ("target_32768x65535", (32768, 65535)),
]

_MODULE_CACHE = {}
_MODEL_CACHE = {}


def _load(key, path):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    path = {
        "baseline1": INPUT_FILE,
        "baseline2": BASE_FILE,
        "optimized": OPT_FILE
    }[key]
    mod = _load(key, path)
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    _MODEL_CACHE[key] = mod.ModelNew(*init).to(device="npu").eval()
    return _MODEL_CACHE[key]


def _make_inputs(shape, device="npu"):
    torch.manual_seed(0)
    return [torch.rand(shape, device=device, dtype=torch.float32)]


def _run_torch_ref(x):
    xf = x.to(torch.float32)
    denom = torch.linalg.vector_norm(xf, ord=2, dim=1, keepdim=True)
    return xf / denom


def _run_provider(key, x):
    return _model(key)(x)


def _max_abs(a, b):
    return (a - b).abs().max().item()


def _baseline_grid_ok(x):
    return x.shape[0] <= 65535


def _optimized_grid_ok(x):
    return True


def unit_test():
    ok = True
    providers = [("baseline1", "Baseline Triton1")]
    if BASE_FILE.exists():
        providers.append(("baseline2", "Baseline Triton2"))
    providers.append(("optimized", "Optimized Triton"))
    for label, shape in _BENCH_SHAPES:
        x = _make_inputs(shape)[0]
        ref = _run_torch_ref(x)
        print(f"TEST PyTorch / ACL {label} PASS")
        for key, name in providers:
            try:
                if key in ("baseline1",
                           "baseline2") and not _baseline_grid_ok(x):
                    print(
                        f"TEST {key} {label} SKIP coreDim_guard rows={x.shape[0]}"
                    )
                    continue
                if key == "optimized" and not _optimized_grid_ok(x):
                    ok = False
                    print(f"TEST {key} {label} MISMATCH coreDim_guard")
                    continue
                y = _run_provider(key, x)
                torch.npu.synchronize()
                torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)
                print(
                    f"TEST {key} {label} PASS max_abs={_max_abs(y, ref):.6e}")
            except Exception as e:
                if key == "optimized":
                    ok = False
                    print(
                        f"TEST {key} {label} MISMATCH {type(e).__name__}: {e}")
                else:
                    print(f"TEST {key} {label} INFO {type(e).__name__}: {e}")
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _manual_bench(fn, warmup=3, rep=10):
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / rep


def bench_provider(provider, label):
    shape = dict((item, s) for item, s in _BENCH_SHAPES)[label]
    x = _make_inputs(shape)[0]
    try:
        if provider in ("baseline1", "baseline2") and not _baseline_grid_ok(x):
            return float("inf")
        if provider == "optimized" and not _optimized_grid_ok(x):
            return float("inf")
        if provider == "torch":

            def fn():
                return _run_torch_ref(x)
        else:
            ref = _run_torch_ref(x)
            y = _run_provider(provider, x)
            torch.npu.synchronize()
            torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)

            def fn():
                return _run_provider(provider, x)

        try:
            return triton.testing.do_bench(fn,
                                           warmup=5,
                                           rep=20,
                                           return_mode="mean")
        except Exception:
            return _manual_bench(fn)
    except Exception as e:
        print(f"INFO bench {provider} {label}: {type(e).__name__}: {e}")
        return float("inf")


_line_vals = ["torch", "baseline1"]
_line_names = ["PyTorch / ACL", "Baseline Triton1"]
_styles = [("green", "-"), ("blue", "--")]
if BASE_FILE.exists():
    _line_vals.append("baseline2")
    _line_names.append("Baseline Triton2")
    _styles.append(("black", "--"))
_line_vals.append("optimized")
_line_names.append("Optimized Triton")
_styles.append(("red", "-"))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[x[0] for x in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_line_vals,
        line_names=_line_names,
        styles=_styles,
        ylabel="ms",
        plot_name="l2norm_rowwise",
        args={},
    ))
def benchmark(label, provider):
    return bench_provider(provider, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if not args.test and not args.bench:
        args.test = args.bench = True
    if args.test:
        unit_test()
    if args.bench:
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
