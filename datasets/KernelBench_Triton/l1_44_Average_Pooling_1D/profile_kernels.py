import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "44_Average_Pooling_1D.py"
OPT_FILE = ROOT / "opt_44_Average_Pooling_1D.py"
BASE2_FILES = sorted(ROOT.glob("base_*.py"))
BASE2_FILE = BASE2_FILES[0] if BASE2_FILES else None

KERNEL_SIZE = 8
STRIDE = 1
PADDING = 4
_BENCH_SHAPES = [
    ("direct_small", 2, 4, 1024),
    ("persistent_cover", 256, 512, 33),
    ("target", 64, 128, 65536),
]
_PROVIDER_KEYS = ["torch", "baseline1"
                  ] + (["baseline2"] if BASE2_FILE else []) + ["optimized"]
_PROVIDER_NAMES = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1" if BASE2_FILE else "Baseline Triton",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}
_MODULE_CACHE = {}
_MODEL_CACHE = {}
_SKIP_REASON = {}


def _source_has_unsupported(path: Path) -> str:
    try:
        text = path.read_text()
    except Exception as exc:
        return f"read_error:{exc}"
    if 'cache_modifier=".cg"' in text or "cache_modifier='.cg'" in text:
        return "unsupported_cache_modifier_cg"
    return ""


def _load(key: str, path: Path):
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(f"k_l1_44_{key}_{path.stem}",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def _model(key: str):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if key == "baseline1":
        reason = _source_has_unsupported(INPUT_FILE)
        if reason:
            _SKIP_REASON[key] = reason
            return None
        mod = _load(key, INPUT_FILE)
    elif key == "baseline2":
        if BASE2_FILE is None:
            _SKIP_REASON[key] = "missing_base2"
            return None
        reason = _source_has_unsupported(BASE2_FILE)
        if reason:
            _SKIP_REASON[key] = reason
            return None
        mod = _load(key, BASE2_FILE)
    elif key == "optimized":
        mod = _load(key, OPT_FILE)
    else:
        return None
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    model = mod.ModelNew(*init).to(device="npu").eval()
    _MODEL_CACHE[key] = model
    return model


def _make_input(label: str):
    _, b, c, length = next(s for s in _BENCH_SHAPES if s[0] == label)
    torch.manual_seed(0)
    return torch.rand((b, c, length), device="npu", dtype=torch.float32)


def _run_torch_ref(x):
    return F.avg_pool1d(x,
                        KERNEL_SIZE,
                        STRIDE,
                        PADDING,
                        ceil_mode=False,
                        count_include_pad=True)


def _run_provider(key: str, x):
    if key == "torch":
        return _run_torch_ref(x)
    model = _model(key)
    if model is None:
        raise RuntimeError(_SKIP_REASON.get(key, "provider_unavailable"))
    return model(x)


def _sync():
    torch.npu.synchronize()


def _timed(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
        _sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
        _sync()
    return (time.perf_counter() - t0) * 1000.0 / rep


def unit_test():
    ok = True
    for label, _, _, _ in _BENCH_SHAPES:
        x = _make_input(label)
        ref = _run_torch_ref(x)
        _sync()
        for key in _PROVIDER_KEYS:
            _PROVIDER_NAMES[key]
            if key == "torch":
                print(f"TEST torch {label} PASS")
                continue
            try:
                y = _run_provider(key, x)
                _sync()
                max_abs = (
                    y.float() -
                    ref.float()).abs().max().item() if ref.numel() else 0.0
                if torch.allclose(y.float(), ref.float(), rtol=1e-3,
                                  atol=1e-3):
                    print(f"TEST {key} {label} PASS max_abs={max_abs:.6g}")
                else:
                    print(f"TEST {key} {label} MISMATCH max_abs={max_abs:.6g}")
                    if key == "optimized":
                        ok = False
            except Exception as exc:
                tag = "SKIP" if key != "optimized" else "FAIL"
                print(f"TEST {key} {label} {tag} {type(exc).__name__}:{exc}")
                if key == "optimized":
                    ok = False
        del x, ref
        torch.npu.empty_cache()
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDER_KEYS,
        line_names=[_PROVIDER_NAMES[k] for k in _PROVIDER_KEYS],
        styles=[("black", "-"), ("blue", "-"),
                ("green", "-")] + ([("red", "-")] if BASE2_FILE else []),
        ylabel="ms",
        plot_name="avgpool1d_l1_44",
        args={},
    ))
def bench(label, provider):
    x = _make_input(label)
    try:
        if provider not in ("torch", "optimized") and _source_has_unsupported(
                INPUT_FILE if provider == "baseline1" else BASE2_FILE):
            print(
                f"INFO {provider} {label} INF {_source_has_unsupported(INPUT_FILE if provider == 'baseline1' else BASE2_FILE)}"
            )
            return float("inf")
        y = _run_provider(provider, x)
        _sync()
        del y
        ms = _timed(lambda: _run_provider(provider, x))  # noqa: F821
        return ms
    except Exception as exc:
        print(f"INFO {provider} {label} INF {type(exc).__name__}:{exc}")
        return float("inf")
    finally:
        del x
        torch.npu.empty_cache()


def run_bench():
    bench.run(print_data=True, show_plots=False)


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
