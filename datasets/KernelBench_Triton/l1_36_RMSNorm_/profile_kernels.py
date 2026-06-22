import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton

ROOT = Path(__file__).resolve().parent


def _load(name, file):
    path = ROOT / file
    spec = importlib.util.spec_from_file_location(f"k_{name}_{path.stem}",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mods = {
    "baseline1": _load("baseline1", "36_RMSNorm_.py"),
    "baseline2": _load("baseline2", "base_36_RMSNorm_.py"),
    "optimized": _load("optimized", "opt_36_RMSNorm_.py"),
}

_MODEL_CACHE = {}
_BENCH_SHAPES = [
    ("small", 2, 64, 16, 16),
    ("irregular", 3, 64, 19, 17),
    ("medium", 8, 64, 128, 128),
    ("target", 112, 64, 512, 512),
]


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else [64]
    if init == [()]:
        init = []
    return init


def _model(key, device="npu"):
    ck = (key, device)
    if ck not in _MODEL_CACHE:
        mod = mods[key]
        _MODEL_CACHE[ck] = mod.ModelNew(*_init_args(mod)).to(
            device=device).eval()
    return _MODEL_CACHE[ck]


class TorchRMSNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(
            torch.mean(x.float() * x.float(), dim=1, keepdim=True) + self.eps)


_TORCH_MODEL = None


def _torch_model(device="npu"):
    global _TORCH_MODEL
    if _TORCH_MODEL is None:
        _TORCH_MODEL = TorchRMSNorm(64).to(device=device).eval()
    return _TORCH_MODEL


def _make_inputs(label, B, C, H, W, device="npu"):
    torch.manual_seed(0)
    x = torch.rand((B, C, H, W), device=device, dtype=torch.float32)
    return [x]


def _run_torch_ref(x):
    return _torch_model(x.device.type)(x)


def _run_provider(key, x):
    # Baseline1 launches one program per B*H*W row; avoid Ascend coreDim > 65535
    # because a failed launch poisons the process and invalidates later providers.
    if key == "baseline1" and x.dim(
    ) == 4 and x.shape[0] * x.shape[2] * x.shape[3] > 65535:
        raise RuntimeError(
            "SKIP coreDim_guard: baseline1 grid B*H*W exceeds 65535")
    if key == "baseline2" and x.dim() == 4 and x.shape[0] * triton.cdiv(
            x.shape[2] * x.shape[3], 128) > 65535:
        raise RuntimeError(
            "SKIP coreDim_guard: baseline2 block grid exceeds 65535")
    return _model(key, x.device.type)(x)


def _max_abs_rel(a, b):
    diff = (a - b).abs()
    max_abs = diff.max().item()
    denom = b.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()
    return max_abs, max_rel


def unit_test():
    ok_opt = True
    for shape in _BENCH_SHAPES:
        label, B, C, H, W = shape
        x, = _make_inputs(*shape)
        ref = _run_torch_ref(x)
        for key, display in [("baseline1", "baseline1"),
                             ("baseline2", "baseline2"),
                             ("optimized", "optimized")]:
            try:
                y = _run_provider(key, x)
                torch.npu.synchronize()
                max_abs, max_rel = _max_abs_rel(y, ref)
                passed = (max_abs <= 1e-3) and (max_rel <= 1e-3)
                print(
                    f"TEST {display} {label} {'PASS' if passed else 'MISMATCH'} max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
                )
                if key == "optimized" and not passed:
                    ok_opt = False
            except Exception as e:
                tag = "FAIL" if key == "optimized" else "INFO"
                print(f"TEST {display} {label} {tag} {type(e).__name__}: {e}")
                if key == "optimized":
                    ok_opt = False
    print("UNIT_TEST PASS" if ok_opt else "UNIT_TEST_FAILED")
    return ok_opt


def _bench_one(provider, shape):
    label, B, C, H, W = shape
    x, = _make_inputs(*shape)
    if provider == "torch":

        def fn():
            return _run_torch_ref(x)
    else:

        def fn():
            return _run_provider(provider, x)

    try:
        return triton.testing.do_bench(fn,
                                       warmup=5,
                                       rep=20,
                                       return_mode="mean")
    except Exception as e:
        print(f"INFO bench {provider} {label} {type(e).__name__}: {e}")
        try:
            for _ in range(3):
                fn()
                torch.npu.synchronize()
            t0 = time.perf_counter()
            reps = 10
            for _ in range(reps):
                fn()
                torch.npu.synchronize()
            return (time.perf_counter() - t0) * 1000.0 / reps
        except Exception as e2:
            print(
                f"INFO bench {provider} {label} INF {type(e2).__name__}: {e2}")
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
        styles=[("blue", "-"), ("green", "-"), ("black", "--"), ("red", "-")],
        ylabel="ms",
        plot_name="rmsnorm-performance",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    return _bench_one(provider, shape)


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
