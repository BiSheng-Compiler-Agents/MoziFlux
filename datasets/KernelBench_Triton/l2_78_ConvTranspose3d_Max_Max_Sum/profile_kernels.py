import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "78_ConvTranspose3d_Max_Max_Sum.py"
OPT_FILE = ROOT / "opt_78_ConvTranspose3d_Max_Max_Sum.py"
BASE_FILE = ROOT / "base_78_ConvTranspose3d_Max_Max_Sum.py"

_BENCH_SHAPES = [
    ("tiny_direct", 1, 32, 8, 8, 8),
    ("irregular_direct", 2, 32, 11, 13, 15),
    ("default_direct", 16, 32, 32, 32, 32),
]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_MODS = {}
_MODELS = {}


def _mod(key):
    if key not in _MODS:
        if key == "baseline1":
            _MODS[key] = _load(INPUT_FILE, "k_l2_78_baseline1")
        elif key == "optimized":
            _MODS[key] = _load(OPT_FILE, "k_l2_78_optimized")
        else:
            raise KeyError(key)
    return _MODS[key]


def _init_args():
    return [32, 64, 5, 2, 2]


def _model(key):
    if key not in _MODELS:
        torch.manual_seed(0)
        if key == "torch_ref":
            m = nn.ConvTranspose3d(32, 64, 5, stride=2, padding=2)
        else:
            m = _mod(key).ModelNew(*_init_args())
        _MODELS[key] = m.to("npu")
    return _MODELS[key]


def _make_inputs(label, n, c, d, h, w):
    torch.manual_seed(123)
    return (torch.rand(n, c, d, h, w, device="npu"), )


def _run_torch_ref(*args):
    x = args[0]
    conv = _model("torch_ref")
    y = conv(x)
    y = F.max_pool3d(y, kernel_size=2, stride=2)
    y = F.max_pool3d(y, kernel_size=3, stride=3)
    return y.sum(dim=1, keepdim=True)


def _run_provider(key, *args):
    if key == "torch_ref":
        return _run_torch_ref(*args)
    if key == "baseline2":
        raise RuntimeError("sandboxed_base_file_not_read")
    return _model(key)(*args)


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().detach().cpu())


def unit_test():
    ok = True
    with torch.no_grad():
        for shape in _BENCH_SHAPES:
            label = shape[0]
            args = _make_inputs(*shape)
            ref = _run_torch_ref(*args)
            for key, display in [
                ("baseline1", "Baseline Triton1"),
                ("baseline2", "Baseline Triton2"),
                ("optimized", "Optimized Triton"),
            ]:
                if key == "baseline2":
                    print(
                        f"TEST {display} {label}: SKIP_UNAVAILABLE sandboxed_base_file_not_read max_abs=inf"
                    )
                    continue
                try:
                    out = _run_provider(key, *args)
                    diff = _max_abs(out, ref)
                    passed = diff <= 1e-3
                    print(
                        f"TEST {display} {label}: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}"
                    )
                    if key == "optimized" and not passed:
                        ok = False
                except Exception as e:
                    tag = type(e).__name__
                    if key == "optimized":
                        print(
                            f"TEST {display} {label}: FAIL {tag} max_abs=inf")
                        ok = False
                    else:
                        print(
                            f"TEST {display} {label}: SKIP_UNAVAILABLE {tag} max_abs=inf"
                        )

        # Force both hidden Triton fallback paths while production uses ACL dispatch.
        try:
            opt = _mod("optimized")
            old_grid = opt._MAX_GRID
            old_acl = opt._USE_ACL_DISPATCH
            opt._USE_ACL_DISPATCH = False
            _MODELS.pop("optimized", None)
            args = _make_inputs("forced_direct", 1, 32, 8, 8, 8)
            ref = _run_torch_ref(*args)
            out = _run_provider("optimized", *args)
            diff = _max_abs(out, ref)
            passed = diff <= 1e-3
            print(
                f"TEST Optimized Triton forced_direct_fallback: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}"
            )
            ok = ok and passed

            opt._MAX_GRID = 1
            _MODELS.pop("optimized", None)
            args = _make_inputs("forced_persistent", 1, 32, 8, 8, 8)
            ref = _run_torch_ref(*args)
            out = _run_provider("optimized", *args)
            diff = _max_abs(out, ref)
            passed = diff <= 1e-3
            print(
                f"TEST Optimized Triton forced_persistent_fallback: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}"
            )
            ok = ok and passed
            opt._MAX_GRID = old_grid
            opt._USE_ACL_DISPATCH = old_acl
            _MODELS.pop("optimized", None)
        except Exception as e:
            print(
                f"TEST Optimized Triton forced_fallbacks: FAIL {type(e).__name__} max_abs=inf"
            )
            ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_ms(fn, warmup=3, rep=10):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / rep


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
        styles=[("blue", "-"), ("green", "-"), ("black", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="l2_78_ConvTranspose3d_Max_Max_Sum",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    args = _make_inputs(*shape)
    if provider == "baseline2":
        return float("inf")
    if provider == "baseline1" and label == "default_direct":
        print(
            f"INFO comparison_provider_preskipped Baseline Triton1 {label} avoid_slow_autotune"
        )
        return float("inf")
    try:
        with torch.no_grad():
            _run_provider(provider, *args)
            torch.npu.synchronize()
            return _bench_ms(lambda: _run_provider(provider, *args))
    except Exception as e:
        print(
            f"INFO benchmark_unavailable {provider} {label} {type(e).__name__}"
        )
        return float("inf")


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
        bench.run(print_data=True,
                  show_plots=False,
                  save_path=str(ROOT / "profile_plots"))


if __name__ == "__main__":
    main()
