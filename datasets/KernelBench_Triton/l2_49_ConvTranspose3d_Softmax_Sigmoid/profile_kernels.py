import argparse
import importlib.util
import pathlib
import sys
import time
import traceback

import torch
import torch.nn as nn
import triton

ROOT = pathlib.Path(__file__).resolve().parent
INPUT_FILE = ROOT / "49_ConvTranspose3d_Softmax_Sigmoid.py"
BASE_FILE = ROOT / "base_49_ConvTranspose3d_Softmax_Sigmoid.py"
OPT_FILE = ROOT / "opt_49_ConvTranspose3d_Softmax_Sigmoid.py"

_BENCH_SHAPES = [
    # label, batch, in_ch, out_ch, D, H, W, path
    ("tiny_triton_path", 1, 4, 8, 2, 2, 2, "triton"),
    ("small_triton_path", 1, 8, 64, 4, 4, 4, "triton"),
    ("default_acl_path", 16, 32, 64, 16, 32, 32, "acl"),
]

_MODEL_CACHE = {}
_MODULE_CACHE = {}


def _load(path, name):
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[name] = mod
    return mod


def _init_args(batch, in_ch, out_ch, d, h, w):
    return [in_ch, out_ch, 3, 2, 1, 1]


def _make_inputs(batch, in_ch, out_ch, d, h, w, dtype=torch.float32):
    torch.manual_seed(123)
    x = torch.rand(batch, in_ch, d, h, w, dtype=dtype, device="npu")
    return [x]


class TorchRef(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, bias=True
        )
    def forward(self, x):
        x = self.conv_transpose(x)
        return torch.sigmoid(torch.softmax(x, dim=1))


def _model(key, batch, in_ch, out_ch, d, h, w):
    cache_key = (key, in_ch, out_ch)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    torch.manual_seed(0)
    args = _init_args(batch, in_ch, out_ch, d, h, w)
    if key == "torch":
        m = TorchRef(*args).npu().eval()
    else:
        path = {"baseline1": INPUT_FILE, "baseline2": BASE_FILE, "opt": OPT_FILE}[key]
        if key == "baseline2" and not path.exists():
            _MODEL_CACHE[cache_key] = None
            return None
        mod = _load(path, f"k_{key}_{path.stem}")
        m = mod.ModelNew(*args).npu().eval()
    _MODEL_CACHE[cache_key] = m
    return m


def _run_torch_ref(x, batch, in_ch, out_ch, d, h, w):
    return _model("torch", batch, in_ch, out_ch, d, h, w)(x)


def _provider_key(provider):
    return {
        "PyTorch / ACL": "torch",
        "Baseline Triton1": "baseline1",
        "Baseline Triton2": "baseline2",
        "Optimized Triton": "opt",
    }[provider]


def _safe_provider_call(provider, x, batch, in_ch, out_ch, d, h, w, for_bench=False):
    key = _provider_key(provider)
    if key == "torch":
        return _run_torch_ref(x, batch, in_ch, out_ch, d, h, w)
    if key == "baseline2" and not BASE_FILE.exists():
        raise RuntimeError("baseline2_unavailable")
    # The editable baseline launches one program per N*Dout*Hout*Wout row, which
    # exceeds Ascend's grid cap on the default target and poisons the NPU context.
    dout = (d - 1) * 2 - 2 * 1 + 3 + 1
    hout = (h - 1) * 2 - 2 * 1 + 3 + 1
    wout = (w - 1) * 2 - 2 * 1 + 3 + 1
    total_rows = batch * dout * hout * wout
    if key in ("baseline1", "baseline2") and total_rows > 65535:
        raise RuntimeError("grid_guard_preskip")
    m = _model(key, batch, in_ch, out_ch, d, h, w)
    if m is None:
        raise RuntimeError("baseline2_unavailable")
    return m(x)


def _sync():
    torch.npu.synchronize()


def _bench_ms(fn):
    if hasattr(triton.testing, "do_bench"):
        return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="mean")
    for _ in range(10):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(100):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / 100.0


def unit_test():
    ok = True
    for label, batch, in_ch, out_ch, d, h, w, path in _BENCH_SHAPES:
        inputs = _make_inputs(batch, in_ch, out_ch, d, h, w)
        with torch.no_grad():
            ref = _run_torch_ref(inputs[0], batch, in_ch, out_ch, d, h, w)
            for provider in ["Baseline Triton1", "Baseline Triton2", "Optimized Triton"]:
                try:
                    out = _safe_provider_call(provider, inputs[0], batch, in_ch, out_ch, d, h, w)
                    _sync()
                    max_diff = (out - ref).abs().max().item()
                    tol = 1e-3 if out.dtype in (torch.float16, torch.bfloat16) else 1e-4
                    if max_diff > tol:
                        print(f"UNIT_CHECK {label} {provider}: mismatch max_diff={max_diff:.6g} tol={tol}")
                        if provider == "Optimized Triton":
                            ok = False
                    else:
                        print(f"UNIT_CHECK {label} {provider}: PASS max_diff={max_diff:.6g}")
                except Exception as exc:
                    name = type(exc).__name__
                    if provider == "Optimized Triton":
                        print(f"UNIT_CHECK {label} {provider}: FAIL {name}")
                        ok = False
                    else:
                        print(f"UNIT_CHECK {label} {provider}: INFO unavailable_or_preskipped {name}")
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="convtranspose3d_softmax_sigmoid",
        args={},
    )
)
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, batch, in_ch, out_ch, d, h, w, _path = shape
    x = _make_inputs(batch, in_ch, out_ch, d, h, w)[0]
    try:
        _safe_provider_call(provider, x, batch, in_ch, out_ch, d, h, w, for_bench=True)
        _sync()
    except Exception as exc:
        print(f"INFO {provider} {label} preskipped_or_unavailable {type(exc).__name__}")
        return float("inf")
    def fn():
        _safe_provider_call(provider, x, batch, in_ch, out_ch, d, h, w, for_bench=True)
    try:
        return _bench_ms(fn)
    except Exception as exc:
        print(f"INFO {provider} {label} timing_unavailable {type(exc).__name__}")
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
        bench.run(print_data=True, show_plots=False, save_path=str(ROOT / "perf_plots"))


if __name__ == "__main__":
    main()
