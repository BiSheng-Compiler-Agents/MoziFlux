import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent


def _load(fname, key):
    path = ROOT / fname
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, key):
    try:
        return _load(fname, key)
    except Exception as exc:
        print(f"INFO provider {key} import_skip {type(exc).__name__}")
        return None


_MODS = {
    "baseline1": _load("93_masked_cumsum.py", "baseline1"),
    "baseline2": _load_optional("base_93_masked_cumsum.py", "baseline2"),
    "optimized": _load("opt_93_masked_cumsum.py", "optimized"),
}
_MODEL_CACHE = {}

_BENCH_SHAPES = [
    ("tiny", 4, 16),
    ("odd", 17, 257),
    ("medium", 512, 1024),
    ("target", 32768, 32768),
]
_PROVIDER_NAMES = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}


def _sync():
    torch.npu.synchronize()


def _get_model(key):
    mod = _MODS[key]
    if mod is None:
        return None
    if key not in _MODEL_CACHE:
        init = mod.get_init_inputs() if hasattr(mod,
                                                "get_init_inputs") else [1]
        if init == [()]:
            init = []
        _MODEL_CACHE[key] = mod.ModelNew(*init).to("npu").eval()
    return _MODEL_CACHE[key]


def _make_inputs(rows, cols):
    torch.manual_seed(0)
    # Match source get_inputs(): torch.rand float32 and boolean randint mask.
    x = torch.rand((rows, cols), device="npu")
    mask = torch.randint(0, 2, (rows, cols), device="npu").bool()
    return x, mask


def _torch_ref(x, mask, dim=1):
    return torch.cumsum(x * mask.to(dtype=x.dtype), dim=dim)


def _should_skip_provider(key, rows, cols):
    if key == "torch" or key == "optimized":
        return False, ""
    if _MODS.get(key) is None:
        return True, "import_unavailable"
    # Keep comparison providers parser-visible, but avoid large static scan compilation/remote timeout.
    if cols > 1024 or rows * math.ceil(cols / 512) > 65535:
        return True, "compile_guard"
    return False, ""


def _run_provider(key, x, mask, rows, cols):
    if key == "torch":
        return _torch_ref(x, mask)
    skip, reason = _should_skip_provider(key, rows, cols)
    if skip:
        raise RuntimeError(f"SKIP_{reason}")
    model = _get_model(key)
    return model(x, mask)


def _close(a, b):
    torch.npu.synchronize()
    return torch.allclose(a, b, rtol=1e-3, atol=1e-3)


def unit_test():
    all_opt_ok = True
    for label, rows, cols in _BENCH_SHAPES:
        x, mask = _make_inputs(rows, cols)
        ref = _torch_ref(x, mask)
        _sync()
        for key in ("baseline1", "baseline2", "optimized"):
            name = key
            skip, reason = _should_skip_provider(key, rows, cols)
            if skip:
                print(f"TEST {name} {label} SKIP {reason}")
                continue
            try:
                out = _run_provider(key, x, mask, rows, cols)
                ok = _close(out, ref)
                if ok:
                    print(f"TEST {name} {label} PASS")
                else:
                    max_diff = (out - ref).abs().max().item()
                    if key == "optimized":
                        all_opt_ok = False
                        print(
                            f"TEST {name} {label} OPT_DIFF max_abs={max_diff:.6g}"
                        )
                    else:
                        print(
                            f"TEST {name} {label} SKIP value_diff max_abs={max_diff:.6g}"
                        )
            except Exception as exc:
                if key == "optimized":
                    all_opt_ok = False
                    print(
                        f"TEST {name} {label} OPT_EXCEPTION {type(exc).__name__}"
                    )
                else:
                    print(f"TEST {name} {label} SKIP runtime_unavailable")
        del x, mask, ref
        torch.npu.empty_cache()
    if all_opt_ok:
        print("UNIT_TEST PASS")
    else:
        print("UNIT_TEST_FAILED")
    return all_opt_ok


def _bench_one(provider, label):
    rows, cols = next((r, c) for lbl, r, c in _BENCH_SHAPES if lbl == label)
    skip, _ = _should_skip_provider(provider, rows, cols)
    if skip:
        return float("inf")
    x, mask = _make_inputs(rows, cols)
    try:

        def fn():
            return _run_provider(provider, x, mask, rows, cols)

        # do_bench returns milliseconds; perf_report handles display directly.
        ms = triton.testing.do_bench(fn, warmup=5, rep=20, return_mode="mean")
        return ms
    except Exception:
        return float("inf")
    finally:
        torch.npu.empty_cache()


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
        styles=[("blue", "-"), ("red", "-"), ("orange", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="masked_cumsum-performance",
        args={},
    ))
def benchmark(label, provider):
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
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
