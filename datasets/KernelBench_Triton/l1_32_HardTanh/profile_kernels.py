import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "32_HardTanh.py"
BASE2_FILE = ROOT / "base_32_HardTanh.py"
OPT_FILE = ROOT / "opt_32_HardTanh.py"

MAX_PROGRAMS = 65535
BASELINE1_BLOCK = 4096
OPT_DIRECT_BLOCK = 4096
OPT_PERSISTENT_BLOCK = 8192
OPT_PERSISTENT_THRESHOLD = MAX_PROGRAMS * OPT_DIRECT_BLOCK

_BENCH_SHAPES = [
    ("direct_1M", 1024, 1024),
    ("direct_irregular", 257, 4096),
    ("original_persistent", 4096, 393216),
]

_MODULES = {}
_MODELS = {}
_SKIP_REASONS = {}


def _load(key, path):
    if key in _MODULES:
        return _MODULES[key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULES[key] = mod
    return mod


def _model(key):
    if key in _MODELS:
        return _MODELS[key]
    path = {
        "baseline1": INPUT_FILE,
        "baseline2": BASE2_FILE,
        "optimized": OPT_FILE
    }[key]
    mod = _load(key, path)
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    _MODELS[key] = mod.ModelNew(*init).to(device="npu").eval()
    return _MODELS[key]


def _numel(shape):
    return int(shape[0]) * int(shape[1])


def _make_inputs(label):
    _, batch, dim = next(row for row in _BENCH_SHAPES if row[0] == label)
    torch.manual_seed(0)
    x = torch.rand((batch, dim), device="npu", dtype=torch.float32)
    # Exercise clamp lower/upper branches while preserving the source distribution style.
    return x * 4.0 - 2.0


def _run_torch_ref(x):
    return torch.clamp(x, min=-1.0, max=1.0)


def _provider_skip_reason(provider, label, x):
    n = x.numel()
    if provider == "baseline1" and triton.cdiv(n,
                                               BASELINE1_BLOCK) > MAX_PROGRAMS:
        return f"baseline1 direct grid would exceed Ascend cap: tiles={triton.cdiv(n, BASELINE1_BLOCK)}"
    if provider == "baseline2" and label == "original_persistent":
        return "read-only baseline2 is not launched on original_persistent to avoid possible coreDim poisoning"
    return None


def _run_provider(provider, x, label=None):
    reason = _provider_skip_reason(provider, label or "", x)
    if reason:
        _SKIP_REASONS[(provider, label)] = reason
        return None
    with torch.no_grad():
        return _model(provider)(x)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _assert_close_or_sample(got, expected, label):
    if label == "original_persistent":
        flat_got = got.reshape(-1)
        flat_exp = expected.reshape(-1)
        n = flat_got.numel()
        idx = torch.tensor([0, 1, 1023, n // 2, n - 1024, n - 2, n - 1],
                           device="npu")
        torch.testing.assert_close(flat_got[idx],
                                   flat_exp[idx],
                                   rtol=1e-3,
                                   atol=1e-3)
    else:
        torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)


def unit_test():
    providers = ["baseline1"]
    if BASE2_FILE.exists():
        providers.append("baseline2")
    providers.append("optimized")
    ok = True
    for label, batch, dim in _BENCH_SHAPES:
        x = _make_inputs(label)
        ref = _run_torch_ref(x)
        _sync()
        for provider in providers:
            try:
                out = _run_provider(provider, x, label)
                if out is None:
                    print(
                        f"TEST {provider} {label} SKIP {_SKIP_REASONS[(provider, label)]}"
                    )
                    continue
                _sync()
                _assert_close_or_sample(out, ref, label)
                path = "persistent" if provider == "optimized" and x.numel(
                ) > OPT_PERSISTENT_THRESHOLD else "direct"
                print(f"TEST {provider} {label} PASS path={path}")
            except Exception as exc:
                if provider in ("baseline2", ):
                    print(
                        f"TEST {provider} {label} INFO {type(exc).__name__}: {exc}"
                    )
                else:
                    print(
                        f"TEST {provider} {label} MISMATCH {type(exc).__name__}: {exc}"
                    )
                    ok = False
        del x, ref
        _sync()
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _time_ms(fn, label):
    # do_bench returns milliseconds on Triton. Keep repetitions modest for the 6 GB original shape.
    warmup = 3 if label == "original_persistent" else 10
    rep = 10 if label == "original_persistent" else 50
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
        start = time.perf_counter()
        for _ in range(rep):
            fn()
            _sync()
        return (time.perf_counter() - start) * 1000.0 / rep


_LINE_VALS = ["torch", "baseline1"] + (["baseline2"] if BASE2_FILE.exists()
                                       else []) + ["optimized"]
_LINE_NAMES = ["PyTorch / ACL", "Baseline Triton1"
               ] + (["Baseline Triton2"]
                    if BASE2_FILE.exists() else []) + ["Optimized Triton"]
_STYLES = [("black", "-"), ("blue", "-"), ("green", "--"),
           ("red", "-")][:len(_LINE_VALS)]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[row[0] for row in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_LINE_VALS,
        line_names=_LINE_NAMES,
        styles=_STYLES,
        ylabel="ms",
        plot_name="hardtanh-performance",
        args={},
    ))
def bench(label, provider):
    x = _make_inputs(label)
    if provider == "torch":

        def fn():
            return _run_torch_ref(x)  # noqa: F821
    else:
        reason = _provider_skip_reason(provider, label, x)
        if reason:
            print(f"INFO {provider} {label} SKIP {reason}")
            return float("inf")

        def fn():
            return _run_provider(provider, x, label)  # noqa: F821

    try:
        ms = _time_ms(fn, label)
        _sync()
        return ms
    except Exception as exc:
        print(f"INFO {provider} {label} INF {type(exc).__name__}: {exc}")
        return float("inf")
    finally:
        del x
        _sync()


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
        bench.run(print_data=True, show_plots=False, save_path=None)


if __name__ == "__main__":
    main()
