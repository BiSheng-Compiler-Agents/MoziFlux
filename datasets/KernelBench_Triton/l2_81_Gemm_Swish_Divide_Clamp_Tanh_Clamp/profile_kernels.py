import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py"
OPT_FILE = HERE / "opt_81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py"
BASE2_EXISTS = (HERE /
                "base_81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py").exists()

_BENCH_SHAPES = [
    ("small_b1", 1, 8192),
    ("medium_b128", 128, 8192),
    ("target_b1024", 1024, 8192),
]
_MODEL_CACHE = {}


def _load(path: Path, key: str):
    name = f"k_{key}_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mod(key):
    if key == "baseline1":
        return _load(INPUT_FILE, key)
    if key == "opt":
        return _load(OPT_FILE, key)
    return None


def _init_args(mod):
    init = mod.get_init_inputs()
    if init == [()]:
        init = []
    return init


def _model(key):
    if key not in _MODEL_CACHE:
        mod = _mod(key)
        torch.manual_seed(0)
        m = mod.ModelNew(*_init_args(mod)).eval().npu()
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


def _make_inputs(batch, features):
    torch.manual_seed(123)
    return torch.rand(batch, features, device="npu")


def _torch_ref(x, model):
    y = F.linear(x, model.gemm.weight, model.gemm.bias)
    y = y * torch.sigmoid(y)
    y = y * 0.5
    y = torch.clamp(y, -1.0, 1.0)
    y = torch.tanh(y)
    y = torch.clamp(y, -1.0, 1.0)
    return y


def _run_provider(provider, x):
    if provider == "torch":
        return _torch_ref(x, _model("opt"))
    if provider == "baseline1":
        return _model("baseline1")(x)
    if provider == "baseline2":
        raise RuntimeError(
            "Baseline Triton2 intentionally not loaded: sandbox forbids reading base_*.py"
        )
    if provider == "opt":
        return _model("opt")(x)
    raise KeyError(provider)


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _time_ms(fn, warmup=10, rep=50):
    try:
        import triton.testing
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(max(1, min(warmup, 3))):
            fn()
        _sync()
        t0 = time.perf_counter()
        reps = max(1, min(rep, 20))
        for _ in range(reps):
            fn()
        _sync()
        return (time.perf_counter() - t0) * 1000.0 / reps


def unit_test():
    ok = True
    providers = ["baseline1", "baseline2", "opt"]
    for label, batch, features in _BENCH_SHAPES:
        x = _make_inputs(batch, features)
        ref = _run_provider("torch", x)
        for provider in providers:
            display = {
                "baseline1": "Baseline Triton1",
                "baseline2": "Baseline Triton2",
                "opt": "Optimized Triton"
            }[provider]
            if provider == "baseline2":
                print(
                    f"TEST {display} {label}: SKIP_UNAVAILABLE sandbox_no_base_read max_abs=inf"
                )
                continue
            try:
                y = _run_provider(provider, x)
                _sync()
                max_abs = (y - ref).abs().max().item()
                passed = math.isfinite(max_abs) and max_abs <= 1e-3
                ok = ok and (passed or provider != "opt")
                print(
                    f"TEST {display} {label}: {'PASS' if passed else 'MISMATCH'} max_abs={max_abs:.6g}"
                )
            except Exception as exc:
                ok = ok and (provider != "opt")
                print(
                    f"TEST {display} {label}: SKIP_UNAVAILABLE {type(exc).__name__} max_abs=inf"
                )

    # Force-test optimized persistent path without allocating >65535 tiles.
    opt_mod = _mod("opt")
    old = opt_mod._MAX_PROGRAMS
    try:
        opt_mod._MAX_PROGRAMS = 1
        x = _make_inputs(2, 8192)
        ref = _torch_ref(x, _model("opt"))
        y = _model("opt")(x)
        _sync()
        max_abs = (y - ref).abs().max().item()
        passed = math.isfinite(max_abs) and max_abs <= 1e-3
        ok = ok and passed
        print(
            f"TEST Optimized Triton forced_persistent: {'PASS' if passed else 'MISMATCH'} max_abs={max_abs:.6g}"
        )
    finally:
        opt_mod._MAX_PROGRAMS = old
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "baseline1", "baseline2", "opt"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="gemm_swish_divide_clamp_tanh_clamp",
        args={},
    ))
def bench(label, provider):
    shape = {s[0]: s for s in _BENCH_SHAPES}[label]
    _, batch, features = shape
    if provider == "baseline2":
        return float("inf")
    try:
        x = _make_inputs(batch, features)
        # Guard custom Triton epilogue grid cap; target shape is well below it.
        if provider in ("baseline1", "opt"):
            n_elems = batch * 8192
            block = 1024 if provider == "baseline1" else 4096
            if triton.cdiv(n_elems, block) > 65535:
                return float("inf")

        def fn():
            return _run_provider(provider, x)

        return _time_ms(fn)
    except Exception as exc:
        print(
            f"INFO bench_unavailable provider={provider} label={label} reason={type(exc).__name__}"
        )
        return float("inf")


def main():
    unit_test()
    try:
        bench.run(print_data=True, show_plots=False, save_path=None)
    except Exception as exc:
        print(f"INFO perf_report_unavailable {type(exc).__name__}")


if __name__ == "__main__":
    main()
