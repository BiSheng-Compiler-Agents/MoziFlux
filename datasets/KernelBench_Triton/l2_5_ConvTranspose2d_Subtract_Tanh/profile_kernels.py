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

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "5_ConvTranspose2d_Subtract_Tanh.py"
OPT_FILE = HERE / "opt_5_ConvTranspose2d_Subtract_Tanh.py"
BASE2_FILE = HERE / "base_5_ConvTranspose2d_Subtract_Tanh.py"  # do not import in this sandbox

_BENCH_SHAPES = [
    ("direct_1x64x16x16", 1, 64, 16, 16),
    ("irregular_2x64x17x19", 2, 64, 17, 19),
    ("default_32x64x256x256", 32, 64, 256, 256),
]

_MODEL_CACHE = {}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _device():
    return "npu"


def _sync():
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _init_inputs():
    return [64, 64, 4, (64, 1, 1)]


def _model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if key == "baseline2":
        _MODEL_CACHE[key] = None
        return None
    path = INPUT_FILE if key == "baseline1" else OPT_FILE
    mod = _load(path, f"k_{key}_{path.stem}")
    torch.manual_seed(0)
    model = mod.ModelNew(*_init_inputs()).to(_device()).eval()
    _MODEL_CACHE[key] = model
    return model


def _make_input(shape):
    label, n, c, h, w = shape
    torch.manual_seed(123)
    return torch.rand(n, c, h, w, device=_device())


class TorchRef(nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv_transpose = nn.ConvTranspose2d(64,
                                                 64,
                                                 4,
                                                 stride=2,
                                                 padding=1,
                                                 output_padding=1)
        self.bias = nn.Parameter(torch.randn((64, 1, 1)))

    def forward(self, x):
        y = F.conv_transpose2d(
            x,
            self.conv_transpose.weight,
            bias=self.conv_transpose.bias,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            dilation=self.conv_transpose.dilation,
            groups=self.conv_transpose.groups,
        )
        return torch.tanh(y - self.bias)


def _torch_ref_model():
    if "torch_ref" not in _MODEL_CACHE:
        _MODEL_CACHE["torch_ref"] = TorchRef().to(_device()).eval()
    return _MODEL_CACHE["torch_ref"]


def _run_torch_ref(x):
    with torch.no_grad():
        return _torch_ref_model()(x)


def _baseline1_would_overflow(shape):
    _, n, _c, h, w = shape
    out_h = (h - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 + 1
    out_w = (w - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 + 1
    elems = n * 64 * out_h * out_w
    return triton.cdiv(elems, 8192) > 65535


def _run_provider(key, x, shape=None):
    if key == "torch_ref":
        return _run_torch_ref(x)
    if key == "baseline2":
        raise RuntimeError(
            "Baseline Triton2 intentionally unavailable: reference file is read-protected by sandbox"
        )
    if key == "baseline1" and shape is not None and _baseline1_would_overflow(
            shape):
        raise RuntimeError(
            "Baseline Triton1 preskipped: direct grid exceeds Ascend FFTS cap")
    with torch.no_grad():
        return _model(key)(x)


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().detach().cpu())


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_input(shape)
        ref = _run_torch_ref(x)
        for key, display in [("baseline1", "Baseline Triton1"),
                             ("baseline2", "Baseline Triton2"),
                             ("optimized", "Optimized Triton")]:
            if key == "baseline2":
                print(f"TEST {display} {label}: SKIP_UNAVAILABLE max_abs=inf")
                continue
            try:
                out = _run_provider(key, x, shape)
                _sync()
                max_abs = _max_abs(out, ref)
                passed = max_abs <= 1e-3
                print(
                    f"TEST {display} {label}: {'PASS' if passed else 'FAIL'} max_abs={max_abs:.6g}"
                )
                if key == "optimized" and not passed:
                    ok = False
            except Exception as e:
                if key == "optimized":
                    print(
                        f"TEST {display} {label}: FAIL {type(e).__name__} max_abs=inf"
                    )
                    ok = False
                else:
                    print(
                        f"TEST {display} {label}: SKIP_UNAVAILABLE {type(e).__name__} max_abs=inf"
                    )

    # Force optimized persistent path on a modest tensor without allocating the huge default.
    try:
        opt = _load(OPT_FILE, "k_opt_force_persistent")
        old = opt._MAX_PROGRAMS
        opt._MAX_PROGRAMS = 1
        torch.manual_seed(0)
        model = opt.ModelNew(*_init_inputs()).to(_device()).eval()
        x = _make_input(_BENCH_SHAPES[1])
        ref = _run_torch_ref(x)
        out = model(x)
        _sync()
        max_abs = _max_abs(out, ref)
        passed = max_abs <= 1e-3
        print(
            f"TEST Optimized Triton forced_persistent: {'PASS' if passed else 'FAIL'} max_abs={max_abs:.6g}"
        )
        ok = ok and passed
        opt._MAX_PROGRAMS = old
    except Exception as e:
        print(
            f"TEST Optimized Triton forced_persistent: FAIL {type(e).__name__} max_abs=inf"
        )
        ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_once(fn, warmup=10, rep=30):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(3):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        _sync()
        return (time.perf_counter() - t0) * 1000.0 / 10.0


def bench_provider(provider, label):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = _make_input(shape)
    if provider == "baseline2":
        print(
            f"INFO benchmark Baseline Triton2 {label}: unavailable_read_protected"
        )
        return float("inf")
    if provider == "baseline1" and _baseline1_would_overflow(shape):
        print(f"INFO benchmark Baseline Triton1 {label}: preskipped_grid_cap")
        return float("inf")
    try:

        def fn():
            return _run_provider(provider, x, shape)

        return _bench_once(fn)
    except Exception as e:
        print(f"INFO benchmark {provider} {label}: {type(e).__name__}")
        return float("inf")


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
        ylabel="Latency (ms)",
        plot_name="convtranspose2d_subtract_tanh_latency",
        args={},
    ))
def benchmark(label, provider):
    return bench_provider(provider, label)


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
