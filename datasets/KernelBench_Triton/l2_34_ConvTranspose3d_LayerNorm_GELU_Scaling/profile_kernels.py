import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "34_ConvTranspose3d_LayerNorm_GELU_Scaling.py"
BASE_FILE = HERE / "base_34_ConvTranspose3d_LayerNorm_GELU_Scaling.py"
OPT_FILE = HERE / "opt_34_ConvTranspose3d_LayerNorm_GELU_Scaling.py"

_BENCH_SHAPES = [
    # Covers optimized direct dispatch (n_tiles <= 65535).
    ("direct_small", 1, 32, 2, 4, 4),
    # Covers optimized ACL grid-cap fallback and the exact get_inputs() regime.
    ("gridcap_default", 32, 32, 16, 32, 32),
]

IN_CHANNELS = 32
OUT_CHANNELS = 64
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
BIAS = True
EPS = 1e-5
SCALING_FACTOR = 1.0
_MAX_PROGRAMS = 65535
_BASELINE_MIN_ROWS_PER_CTA = 8


def _load(path: Path, key: str):
    name = f"k_{key}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_MODULES = {
    "baseline1": _load(INPUT_FILE, "baseline1"),
    "optimized": _load(OPT_FILE, "optimized"),
}
_MODULE_ERRORS = {}
_HAS_BASELINE2 = BASE_FILE.exists()
if _HAS_BASELINE2:
    try:
        _MODULES["baseline2"] = _load(BASE_FILE, "baseline2")
    except Exception as e:
        # base_*.py is read-only comparison material; keep its column visible as inf.
        _MODULE_ERRORS["baseline2"] = type(e).__name__

_MODELS = {}


class TorchRefModel(nn.Module):

    def __init__(
        self,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
        bias=BIAS,
        eps=EPS,
        scaling_factor=SCALING_FACTOR,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        y = self.conv_transpose(x)
        y = y.permute(0, 2, 3, 4, 1).contiguous()
        y = F.layer_norm(y, (y.shape[-1], ), self.layer_norm.weight,
                         self.layer_norm.bias, self.layer_norm.eps)
        y = F.gelu(y, approximate="none") * self.scaling_factor
        return y.permute(0, 4, 1, 2, 3).contiguous()


def _device():
    return "npu" if hasattr(torch,
                            "npu") and torch.npu.is_available() else "cpu"


def _sync():
    if _device() == "npu":
        torch.npu.synchronize()


def _init_args():
    return [
        IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, BIAS, EPS,
        SCALING_FACTOR
    ]


def _model(key: str):
    if key in _MODELS:
        return _MODELS[key]
    torch.manual_seed(0)
    if key == "torch_ref":
        model = TorchRefModel(*_init_args())
    else:
        model = _MODULES[key].ModelNew(*_init_args())
    model.eval().to(_device())
    _MODELS[key] = model
    return model


def _make_input(shape):
    _, b, c, d, h, w = shape
    torch.manual_seed(123)
    return torch.rand((b, c, d, h, w), device=_device(), dtype=torch.float32)


def _run_torch_ref(x):
    with torch.no_grad():
        return _model("torch_ref")(x)


def _baseline_grid_overflows(shape):
    _, b, _, d, h, w = shape
    out_d = (d - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE
    out_h = (h - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE
    out_w = (w - 1) * STRIDE - 2 * PADDING + KERNEL_SIZE
    n_rows = b * out_d * out_h * out_w
    return triton.cdiv(n_rows, _BASELINE_MIN_ROWS_PER_CTA) > _MAX_PROGRAMS


def _run_provider(key, x, shape=None):
    if key in ("baseline1", "baseline2"
               ) and shape is not None and _baseline_grid_overflows(shape):
        raise RuntimeError("preskipped_grid_overflow_coreDim_gt_65535")
    with torch.no_grad():
        return _model(key)(x)


def _bench_ms(fn, warmup=10, rep=30):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(warmup):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        _sync()
        return (time.perf_counter() - t0) * 1000.0 / rep


def _check_close(label, got, ref):
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)
    max_abs = (got - ref).abs().max().item()
    print(f"CHECK {label}: PASS max_abs={max_abs:.6g}")


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_input(shape)
        ref = _run_torch_ref(x)
        for key, pretty in [("baseline1", "Baseline Triton1"),
                            ("baseline2", "Baseline Triton2"),
                            ("optimized", "Optimized Triton")]:
            if key == "baseline2" and not _HAS_BASELINE2:
                continue
            if key in _MODULE_ERRORS:
                print(
                    f"INFO {pretty}/{label}: unavailable_or_preskipped {_MODULE_ERRORS[key]}"
                )
                continue
            if key not in _MODULES:
                continue
            try:
                out = _run_provider(key, x, shape)
                _sync()
                _check_close(f"{pretty}/{label}", out, ref)
            except Exception as e:
                if key == "optimized":
                    ok = False
                    print(f"CHECK {pretty}/{label}: FAIL {type(e).__name__}")
                else:
                    print(
                        f"INFO {pretty}/{label}: unavailable_or_preskipped {type(e).__name__}"
                    )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"]
        if _HAS_BASELINE2 else ["torch_ref", "baseline1", "optimized"],
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ] if _HAS_BASELINE2 else
        ["PyTorch / ACL", "Baseline Triton", "Optimized Triton"],
        styles=[("black", "-"), ("blue", "--"), ("green", "--"),
                ("red", "-")] if _HAS_BASELINE2 else [("black", "-"),
                                                      ("blue", "--"),
                                                      ("red", "-")],
        ylabel="ms",
        plot_name="convtranspose3d_layernorm_gelu_scaling",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = _make_input(shape)
    try:
        if provider == "torch_ref":

            def fn():
                return _run_torch_ref(x)
        else:
            if provider in _MODULE_ERRORS:
                print(
                    f"INFO {provider}/{label}: inf comparison_provider_unavailable {_MODULE_ERRORS[provider]}"
                )
                return float("inf")
            if provider in ("baseline1",
                            "baseline2") and _baseline_grid_overflows(shape):
                print(
                    f"INFO {provider}/{label}: inf comparison_provider_preskipped_grid_overflow"
                )
                return float("inf")

            def fn():
                return _run_provider(provider, x, shape)

        fn()
        _sync()
        return _bench_ms(fn)
    except Exception as e:
        print(f"INFO {provider}/{label}: inf {type(e).__name__}")
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
        benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()
