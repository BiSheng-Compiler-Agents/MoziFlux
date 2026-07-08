import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "48_Conv3d_Scaling_Tanh_Multiply_Sigmoid.py"
OPT_FILE = ROOT / "opt_48_Conv3d_Scaling_Tanh_Multiply_Sigmoid.py"

_BENCH_SHAPES = [
    ("small", 4, 3, 8, 16, 16),
    ("medium", 16, 3, 12, 32, 32),
    ("default", 128, 3, 16, 64, 64),
]
_MODEL_CACHE = {}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_input_mod = None
_opt_mod = None


def _mods():
    global _input_mod, _opt_mod
    if _input_mod is None:
        _input_mod = _load(INPUT_FILE, "k_l2_48_input")
    if _opt_mod is None:
        _opt_mod = _load(OPT_FILE, "k_l2_48_opt")
    return _input_mod, _opt_mod


class TorchReference(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=16,
                 kernel_size=3,
                 scaling_factor=2,
                 bias_shape=(16, 1, 1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor_value = scaling_factor
        self.scaling_factor = nn.Parameter(
            torch.full(bias_shape, float(scaling_factor)))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = x * self.scaling_factor
        x = torch.tanh(x)
        x = x * self.bias
        return torch.sigmoid(x)


def _init_args():
    return [3, 16, 3, 2, (16, 1, 1, 1)]


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    inp, opt = _mods()
    torch.manual_seed(0)
    if key == "torch":
        m = TorchReference(*_init_args()).npu().eval()
    elif key == "baseline":
        m = inp.ModelNew(*_init_args()).npu().eval()
    elif key == "opt":
        m = opt.ModelNew(*_init_args()).npu().eval()
    else:
        raise KeyError(key)
    _MODEL_CACHE[key] = m
    return m


def _make_inputs(label):
    shape = next(s[1:] for s in _BENCH_SHAPES if s[0] == label)
    torch.manual_seed(123)
    return (torch.rand(*shape, device="npu", dtype=torch.float32), )


def _run_torch_ref(x):
    with torch.no_grad():
        return _model("torch")(x)


def _run_provider(provider, x):
    with torch.no_grad():
        if provider == "baseline":
            return _model("baseline")(x)
        if provider == "opt":
            return _model("opt")(x)
        if provider == "torch":
            return _run_torch_ref(x)
        raise KeyError(provider)


def _sync():
    torch.npu.synchronize()


def _max_abs(a, b):
    return (a - b).abs().max().detach().float().cpu().item()


def _dispatch_path_tests():
    """Unit-only coverage for optimized direct/persistent and C=16/generic paths."""
    global _opt_mod
    _, opt = _mods()
    ok = True

    def run_case(name, out_channels, force_persistent):
        nonlocal ok
        shape = (2, 3, 8, 16, 16)
        bias_shape = (out_channels, 1, 1, 1)
        torch.manual_seed(321)
        x = torch.rand(*shape, device="npu", dtype=torch.float32)
        old_max = getattr(opt, "_MAX_GRID", 65535)
        if force_persistent:
            opt._MAX_GRID = 1
        try:
            torch.manual_seed(77)
            ref = TorchReference(3, out_channels, 3, 2,
                                 bias_shape).npu().eval()
            torch.manual_seed(77)
            mod = opt.ModelNew(3, out_channels, 3, 2, bias_shape).npu().eval()
            with torch.no_grad():
                y_ref = ref(x)
                y = mod(x)
            _sync()
            diff = _max_abs(y, y_ref)
            passed = math.isfinite(diff) and diff <= 1e-3
            print(
                f"TEST dispatch {name}: max_abs={diff:.6g} {'PASS' if passed else 'MISMATCH'}"
            )
            ok = ok and passed
        except Exception as e:
            print(f"TEST dispatch {name}: EXCEPTION {type(e).__name__}")
            ok = False
        finally:
            opt._MAX_GRID = old_max

    run_case("C16_direct", 16, False)
    run_case("C16_persistent", 16, True)
    run_case("generic_direct", 8, False)
    run_case("generic_persistent", 8, True)
    return ok


def unit_test():
    ok = True
    for label, *_ in _BENCH_SHAPES:
        x, = _make_inputs(label)
        ref = _run_torch_ref(x)
        _sync()
        for key, name in [("baseline", "Baseline Triton"),
                          ("opt", "Optimized Triton")]:
            try:
                got = _run_provider(key, x)
                _sync()
                diff = _max_abs(got, ref)
                passed = math.isfinite(diff) and diff <= 1e-3
                print(
                    f"TEST {label} {name}: max_abs={diff:.6g} {'PASS' if passed else 'MISMATCH'}"
                )
                ok = ok and passed
            except Exception as e:
                print(f"TEST {label} {name}: EXCEPTION {type(e).__name__}")
                ok = False
    ok = _dispatch_path_tests() and ok
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_one(provider, label):
    x, = _make_inputs(label)
    try:

        def fn():
            return _run_provider(provider, x)

        # do_bench returns milliseconds; perf_report labels the axis.
        return triton.testing.do_bench(fn,
                                       warmup=5,
                                       rep=20,
                                       return_mode="mean")
    except Exception as e:
        print(
            f"INFO bench_unavailable provider={provider} label={label} reason={type(e).__name__}"
        )
        return float("inf")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch", "baseline", "opt"],
        line_names=["PyTorch / ACL", "Baseline Triton", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="conv3d_scaling_tanh_multiply_sigmoid",
        args={},
    ))
def bench(label, provider):
    return _bench_one(provider, label)


def run_bench():
    bench.run(print_data=True, show_plots=False, save_path=None)


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
