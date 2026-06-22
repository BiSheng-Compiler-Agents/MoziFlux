import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "40_LayerNorm.py"
BASE_FILE = HERE / "base_40_LayerNorm.py"
OPT_FILE = HERE / "opt_40_LayerNorm.py"

_BENCH_SHAPES = [
    ("batch1_target_norm", (1, 64, 256, 256), (64, 256, 256)),
    ("target", (16, 64, 256, 256), (64, 256, 256)),
]


def _load(path: Path, key: str):
    if not path.exists():
        return None
    name = f"k_{key}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MODS = {
    "baseline1": _load(INPUT_FILE, "baseline1"),
    "baseline2": _load(BASE_FILE, "baseline2"),
    "optimized": _load(OPT_FILE, "optimized"),
}
_MODEL_CACHE = {}


def _sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def _make_input(shape):
    torch.manual_seed(0)
    return torch.rand(*shape, device="npu", dtype=torch.float32)


def _new_model(provider, norm_shape):
    key = (provider, tuple(norm_shape))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    mod = MODS[provider]
    if mod is None:
        return None
    model = mod.ModelNew(tuple(norm_shape)).to(device="npu").eval()
    # Match the source constructor exactly: gamma=1, beta=0.
    with torch.no_grad():
        if hasattr(model, "weight"):
            model.weight.fill_(1.0)
        if hasattr(model, "bias"):
            model.bias.zero_()
    _MODEL_CACHE[key] = model
    return model


def _run_torch_ref(x, norm_shape):
    weight = torch.ones(tuple(norm_shape),
                        device=x.device,
                        dtype=torch.float32)
    bias = torch.zeros(tuple(norm_shape), device=x.device, dtype=torch.float32)
    return F.layer_norm(x, tuple(norm_shape), weight, bias, 1e-5)


def _run_provider(provider, x, norm_shape):
    model = _new_model(provider, norm_shape)
    if model is None:
        raise RuntimeError(f"provider {provider} missing")
    return model(x)


def _assert_close(provider, label, got, ref):
    diff = (got - ref).abs()
    max_abs = float(diff.max().item())
    denom = ref.abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max().item())
    ok = torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
    status = "PASS" if ok else "MISMATCH"
    print(
        f"TEST {provider} {label} {status} max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
    )
    return ok


def unit_test():
    all_opt_ok = True
    for label, shape, norm_shape in _BENCH_SHAPES:
        x = _make_input(shape)
        ref = _run_torch_ref(x, norm_shape)
        _sync()
        for provider in ("baseline1", "baseline2", "optimized"):
            if MODS[provider] is None:
                print(f"TEST {provider} {label} SKIP missing_provider")
                continue
            try:
                got = _run_provider(provider, x, norm_shape)
                _sync()
                ok = _assert_close(provider, label, got, ref)
                if provider == "optimized" and not ok:
                    all_opt_ok = False
            except Exception as e:
                print(
                    f"TEST {provider} {label} SKIP exception={type(e).__name__}:{str(e)[:160]}"
                )
                if provider == "optimized":
                    all_opt_ok = False
    # Dispatch-path coverage for optimized persistent path: many small rows -> total_tasks > 65535.
    try:
        x = _make_input((70000, 8))
        ref = _run_torch_ref(x, (8, ))
        got = _run_provider("optimized", x, (8, ))
        _sync()
        ok = _assert_close("optimized", "persistent_dispatch", got, ref)
        all_opt_ok = all_opt_ok and ok
    except Exception as e:
        print(
            f"TEST optimized persistent_dispatch MISMATCH exception={type(e).__name__}:{str(e)[:160]}"
        )
        all_opt_ok = False
    print("UNIT_TEST PASS" if all_opt_ok else "UNIT_TEST_FAILED")
    return all_opt_ok


def _bench_one(provider, label, shape, norm_shape):
    x = _make_input(shape)
    if provider == "torch":

        def fn():
            return _run_torch_ref(x, norm_shape)
    else:
        if MODS.get(provider) is None:
            return float("inf")
        # Guard direct grids that would exceed Ascend FFTS cap. Optimized has persistent fallback.
        total_tiles = math.prod(shape) // math.prod(norm_shape) * triton.cdiv(
            math.prod(norm_shape), 8192)
        if provider in ("baseline1", "baseline2") and total_tiles > 65535:
            return float("inf")

        def fn():
            return _run_provider(provider, x, norm_shape)

    try:
        for _ in range(5):
            fn()
            _sync()
        if hasattr(triton.testing, "do_bench"):
            return triton.testing.do_bench(fn,
                                           warmup=5,
                                           rep=20,
                                           return_mode="mean")
        start = time.perf_counter()
        reps = 20
        for _ in range(reps):
            fn()
            _sync()
        return (time.perf_counter() - start) * 1000.0 / reps
    except Exception as e:
        print(
            f"INFO bench {provider} {label} inf exception={type(e).__name__}:{str(e)[:120]}"
        )
        return float("inf")


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
        styles=[("black", "-"), ("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="layernorm-performance",
        args={},
    ))
def bench(label, provider):
    shape, norm_shape = None, None
    for item_label, item_shape, item_norm in _BENCH_SHAPES:
        if item_label == label:
            shape, norm_shape = item_shape, item_norm
            break
    return _bench_one(provider, label, shape, norm_shape)


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
