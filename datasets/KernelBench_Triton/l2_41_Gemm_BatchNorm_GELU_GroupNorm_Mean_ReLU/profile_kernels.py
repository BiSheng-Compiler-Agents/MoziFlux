import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU.py"
BASE_FILE = ROOT / "base_41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU.py"
OPT_FILE = ROOT / "opt_41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU.py"

_BENCH_SHAPES = [
    ("default_128", 128, 512, 1024, 8),
    ("small_7", 7, 512, 1024, 8),
    ("irregular_257", 257, 512, 1024, 8),
]

_MODULES = {}
_MODELS = {}


def _load(path: Path, key: str):
    if key in _MODULES:
        return _MODULES[key]
    try:
        spec = importlib.util.spec_from_file_location(
            f"k_l2_41_{key}_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MODULES[key] = mod
        return mod
    except BaseException as exc:  # keep comparison providers visible even if unavailable
        print(f"INFO provider_unavailable {key}: {type(exc).__name__}")
        _MODULES[key] = None
        return None


class TorchRef(nn.Module):

    def __init__(self, in_features, out_features, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        y = self.gemm(x)
        y = self.batch_norm(y)
        y = F.gelu(y, approximate="none")
        y = self.group_norm(y)
        y = y.mean(dim=1, keepdim=True)
        return F.relu(y)


def _init_args(mod, fallback):
    if mod is None or not hasattr(mod, "get_init_inputs"):
        return list(fallback)
    init = mod.get_init_inputs()
    if init == [()]:
        return []
    return list(init)


def _model(key, in_features=512, out_features=1024, num_groups=8):
    cache_key = (key, in_features, out_features, num_groups)
    if cache_key in _MODELS:
        return _MODELS[cache_key]
    torch.manual_seed(0)
    if key == "torch_ref":
        model = TorchRef(in_features, out_features, num_groups)
    else:
        path = {"input": INPUT_FILE, "base": BASE_FILE, "opt": OPT_FILE}[key]
        mod = _load(path, key)
        if mod is None:
            _MODELS[cache_key] = None
            return None
        args = _init_args(mod, [in_features, out_features, num_groups])
        model = mod.ModelNew(*args)
    model = model.to("npu")
    model.eval(
    )  # deterministic BatchNorm path for correctness and benchmarking
    _MODELS[cache_key] = model
    return model


def _make_inputs(batch, in_features):
    torch.manual_seed(123)
    return torch.randn(batch, in_features, device="npu")


def _run_torch_ref(x, in_features=512, out_features=1024, num_groups=8):
    with torch.no_grad():
        return _model("torch_ref", in_features, out_features, num_groups)(x)


def _run_provider(key, x, in_features=512, out_features=1024, num_groups=8):
    model = _model(key, in_features, out_features, num_groups)
    if model is None:
        return None
    with torch.no_grad():
        return model(x)


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().detach().item()


def unit_test():
    ok = True
    providers = [("input", "Baseline Triton1"), ("base", "Baseline Triton2"),
                 ("opt", "Optimized Triton")]
    for label, batch, in_features, out_features, num_groups in _BENCH_SHAPES:
        x = _make_inputs(batch, in_features)
        ref = _run_torch_ref(x, in_features, out_features, num_groups)
        torch.npu.synchronize()
        for key, name in providers:
            y = _run_provider(key, x, in_features, out_features, num_groups)
            if y is None:
                print(f"INFO unit {name} {label}: unavailable")
                continue
            torch.npu.synchronize()
            diff = _max_abs(y, ref)
            passed = diff <= 1e-3
            print(
                f"UNIT {name} {label}: max_abs={diff:.6g} {'PASS' if passed else 'MISMATCH'}"
            )
            if key == "opt" and not passed:
                ok = False

    # Force the optimized persistent dispatch on a small tensor by temporarily
    # lowering its grid cap; this covers the persistent path without allocating
    # a >65k-tile production tensor.
    opt_mod = _load(OPT_FILE, "opt")
    if opt_mod is not None:
        old_max = opt_mod._MAX_GRID
        opt_mod._MAX_GRID = 1
        _MODELS.clear()
        x = _make_inputs(2049, 512)
        ref = _run_torch_ref(x)
        y = _run_provider("opt", x)
        torch.npu.synchronize()
        diff = _max_abs(y, ref)
        passed = diff <= 1e-3
        print(
            f"UNIT Optimized Triton forced_persistent_2049: max_abs={diff:.6g} {'PASS' if passed else 'MISMATCH'}"
        )
        ok = ok and passed
        opt_mod._MAX_GRID = old_max
        _MODELS.clear()
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_ms(fn, warmup=25, rep=200):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(10):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / 50.0


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "input", "base", "opt"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "--"), ("black", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="l2_41_gemm_batchnorm_gelu_groupnorm_mean_relu",
        args={},
    ))
def bench(label, provider):
    shape = {s[0]: s for s in _BENCH_SHAPES}[label]
    _, batch, in_features, out_features, num_groups = shape
    x = _make_inputs(batch, in_features)
    if provider == "torch":

        def fn():
            return _run_torch_ref(x, in_features, out_features, num_groups)
    else:
        key = {"input": "input", "base": "base", "opt": "opt"}[provider]
        if _model(key, in_features, out_features, num_groups) is None:
            print(f"INFO bench provider_unavailable {provider} {label}")
            return float("inf")

        def fn():
            return (_run_provider(key, x, in_features, out_features,
                                  num_groups))

    try:
        return _bench_ms(fn)
    except BaseException as exc:
        print(
            f"INFO bench_unavailable {provider} {label}: {type(exc).__name__}")
        return float("inf")


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
