#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "47_Conv3d_Mish_Tanh.py"
BASE_FILE = HERE / "base_47_Conv3d_Mish_Tanh.py"
OPT_FILE = HERE / "opt_47_Conv3d_Mish_Tanh.py"

_BENCH_CASES = {
    "tiny":
    dict(N=1, in_ch=4, out_ch=8, D=8, H=8, W=8, k=3, stride=1, padding=0),
    "irregular":
    dict(N=2, in_ch=5, out_ch=7, D=9, H=11, W=13, k=3, stride=1, padding=0),
    "persistent_unit":
    dict(N=2, in_ch=8, out_ch=16, D=12, H=12, W=12, k=3, stride=1, padding=0),
    "default":
    dict(N=16, in_ch=32, out_ch=64, D=32, H=64, W=64, k=3, stride=1,
         padding=0),
}
_BENCH_LABELS = ["tiny", "irregular", "default"]
_PROVIDERS = ["torch", "input", "base", "opt"]
_PROVIDER_NAMES = {
    "torch": "PyTorch / ACL",
    "input": "Baseline Triton1",
    "base": "Baseline Triton2",
    "opt": "Optimized Triton",
}
_mod_cache = {}
_model_cache = {}
_input_cache = {}


def _device():
    return torch.device("npu")


def _sync():
    if torch.npu.is_available():
        torch.npu.synchronize()


def _load(path: Path, key: str):
    if key in _mod_cache:
        return _mod_cache[key]
    try:
        spec = importlib.util.spec_from_file_location(
            f"k_l2_47_{key}_{path.stem}", str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _mod_cache[key] = mod
        return mod
    except Exception as exc:
        _mod_cache[key] = exc
        return exc


def _make_inputs(label):
    if label in _input_cache:
        return _input_cache[label]
    c = _BENCH_CASES[label]
    torch.manual_seed(123)
    x = torch.rand(c["N"],
                   c["in_ch"],
                   c["D"],
                   c["H"],
                   c["W"],
                   dtype=torch.float32,
                   device=_device())
    _input_cache[label] = (x, )
    return (x, )


class TorchRef(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, x):
        y = self.conv(x)
        return torch.tanh(F.mish(y))


def _model(provider, label):
    key = (provider, label)
    if key in _model_cache:
        return _model_cache[key]
    c = _BENCH_CASES[label]
    args = [c["in_ch"], c["out_ch"], c["k"]]
    kwargs = {"stride": c["stride"], "padding": c["padding"]}
    torch.manual_seed(0)
    if provider == "torch":
        m = TorchRef(*args, **kwargs).to(_device()).eval()
    else:
        path = {
            "input": INPUT_FILE,
            "base": BASE_FILE,
            "opt": OPT_FILE
        }[provider]
        mod = _load(path, provider)
        if isinstance(mod, Exception):
            _model_cache[key] = mod
            return mod
        m = mod.ModelNew(*args, **kwargs).to(_device()).eval()
    _model_cache[key] = m
    return m


def _run_provider(provider, label):
    m = _model(provider, label)
    if isinstance(m, Exception):
        raise RuntimeError(
            f"provider_unavailable:{provider}:{type(m).__name__}")
    x, = _make_inputs(label)
    with torch.no_grad():
        return m(x)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok_opt = True
    for label in _BENCH_LABELS:
        ref = _run_provider("torch", label)
        _sync()
        for provider in ["input", "base", "opt"]:
            try:
                out = _run_provider(provider, label)
                _sync()
                diff = _max_abs(out, ref)
                status = "PASS" if diff <= 3e-3 else "MISMATCH"
                print(
                    f"TEST {status} provider={_PROVIDER_NAMES[provider]} label={label} max_abs={diff:.6g}"
                )
                if provider == "opt" and diff > 3e-3:
                    ok_opt = False
            except Exception as exc:
                print(
                    f"INFO provider_unavailable provider={_PROVIDER_NAMES[provider]} label={label} reason={type(exc).__name__}"
                )
                if provider == "opt":
                    ok_opt = False
    # Forced persistent dispatch without huge allocation.
    try:
        mod = _load(OPT_FILE, "opt")
        old = getattr(mod, "_MAX_GRID", 65535)
        mod._MAX_GRID = 1
        _model_cache.pop(("opt", "persistent_unit"), None)
        ref = _run_provider("torch", "persistent_unit")
        out = _run_provider("opt", "persistent_unit")
        _sync()
        diff = _max_abs(out, ref)
        print(
            f"TEST {'PASS' if diff <= 3e-3 else 'MISMATCH'} provider=Optimized_Triton_forced_persistent label=persistent_unit max_abs={diff:.6g}"
        )
        ok_opt = ok_opt and diff <= 3e-3
        mod._MAX_GRID = old
        _model_cache.pop(("opt", "persistent_unit"), None)
    except Exception as exc:
        print(
            f"INFO forced_persistent_unavailable reason={type(exc).__name__}")
        ok_opt = False
    print("UNIT_TEST PASS" if ok_opt else "UNIT_TEST_FAILED")
    return ok_opt


def _bench_one(provider, label):
    try:
        # Build and compile outside timed function.
        _run_provider(provider, label)
        _sync()

        def fn():
            _run_provider(provider, label)

        return triton.testing.do_bench(fn,
                                       warmup=5,
                                       rep=20,
                                       return_mode="mean")
    except Exception as exc:
        print(
            f"INFO bench_unavailable provider={_PROVIDER_NAMES[provider]} label={label} reason={type(exc).__name__}"
        )
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=_BENCH_LABELS,
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[_PROVIDER_NAMES[p] for p in _PROVIDERS],
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="conv3d_mish_tanh",
        args={},
    ))
def benchmark(label, provider):
    return _bench_one(provider, label)


def run_bench():
    benchmark.run(print_data=True, show_plots=False, save_path=str(HERE))


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
        run_bench()


if __name__ == "__main__":
    main()
