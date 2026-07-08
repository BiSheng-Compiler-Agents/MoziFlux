import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "73_Conv2d_BatchNorm_Scaling.py"
OPT_FILE = ROOT / "opt_73_Conv2d_BatchNorm_Scaling.py"
BASE_FILE = ROOT / "base_73_Conv2d_BatchNorm_Scaling.py"

_BENCH_SHAPES = [
    ("small_direct", 4, 8, 33, 35),
    ("medium_direct", 32, 8, 64, 64),
    ("default", 128, 8, 128, 128),
]
_PROVIDERS = [
    "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"
]
_MODEL_CACHE = {}
_MODULE_CACHE = {}


class TorchRef(nn.Module):

    def __init__(self,
                 in_channels=8,
                 out_channels=64,
                 kernel_size=3,
                 scaling_factor=2.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return self.bn(self.conv(x)) * float(self.scaling_factor)


def _load(path: Path, name: str):
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[name] = mod
    return mod


def _init_args(mod):
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    if init == [()]:
        init = []
    return init


def _make_inputs(label, batch, channels, height, width):
    torch.manual_seed(123)
    x = torch.rand(batch,
                   channels,
                   height,
                   width,
                   device="npu",
                   dtype=torch.float32)
    return (x, )


def _fresh_model(provider):
    torch.manual_seed(0)
    if provider == "PyTorch / ACL":
        m = TorchRef().npu()
    elif provider == "Baseline Triton1":
        mod = _load(INPUT_FILE, "k_input_73_conv_bn_scaling")
        m = mod.ModelNew(*_init_args(mod)).npu()
    elif provider == "Optimized Triton":
        mod = _load(OPT_FILE, "k_opt_73_conv_bn_scaling")
        m = mod.ModelNew(*_init_args(mod)).npu()
    else:
        return None
    m.train()
    return m


def _model(provider, label):
    key = (provider, label)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _fresh_model(provider)
    return _MODEL_CACHE[key]


def _run_provider(provider, label, x):
    if provider == "Baseline Triton2":
        raise RuntimeError(
            "Baseline Triton2 is read-only and skipped by sandbox policy")
    return _model(provider, label)(x)


def _max_abs(a, b):
    return float((a - b).abs().max().detach().cpu())


def unit_test():
    ok = True
    for label, b, c, h, w in _BENCH_SHAPES:
        x = _make_inputs(label, b, c, h, w)[0]
        ref = _fresh_model("PyTorch / ACL")
        with torch.no_grad():
            y_ref = ref(x)
        for provider in _PROVIDERS[1:]:
            if provider == "Baseline Triton2":
                print(
                    f"TEST Baseline Triton2 {label}: SKIP_SANDBOX_DO_NOT_READ max_abs=inf"
                )
                continue
            try:
                m = _fresh_model(provider)
                with torch.no_grad():
                    y = m(x)
                diff = _max_abs(y, y_ref)
                passed = diff <= 1e-3
                print(
                    f"TEST {provider} {label}: {'PASS' if passed else 'FAIL'} max_abs={diff:.6g}"
                )
                ok = ok and (passed or provider == "Baseline Triton1")
            except Exception as exc:
                tag = type(exc).__name__
                print(
                    f"TEST {provider} {label}: {'FAIL' if provider == 'Optimized Triton' else 'SKIP_UNAVAILABLE'} {tag} max_abs=inf"
                )
                if provider == "Optimized Triton":
                    ok = False

    # Force-test both optimized fallback scale dispatch paths on a modest tensor.
    try:
        opt = _load(OPT_FILE, "k_opt_73_conv_bn_scaling")
        x = torch.rand(9000, device="npu", dtype=torch.float32)
        with torch.no_grad():
            y_direct = opt._scale_triton(x, 2.0, force_persistent=False)
            y_persist = opt._scale_triton(x, 2.0, force_persistent=True)
        d0 = _max_abs(y_direct, x * 2.0)
        d1 = _max_abs(y_persist, x * 2.0)
        print(
            f"TEST Optimized Triton scale_direct: {'PASS' if d0 <= 1e-6 else 'FAIL'} max_abs={d0:.6g}"
        )
        print(
            f"TEST Optimized Triton scale_persistent: {'PASS' if d1 <= 1e-6 else 'FAIL'} max_abs={d1:.6g}"
        )
        ok = ok and d0 <= 1e-6 and d1 <= 1e-6
    except Exception as exc:
        print(
            f"TEST Optimized Triton scale_paths: FAIL {type(exc).__name__} max_abs=inf"
        )
        ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_ms(fn, warmup=10, rep=50):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(max(1, rep // 5)):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - start) * 1000.0 / max(1, rep // 5)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=_PROVIDERS,
        styles=[("black", "-"), ("blue", "-"), ("red", "--"), ("green", "-")],
        ylabel="ms",
        plot_name="conv2d_batchnorm_scaling",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    if provider == "Baseline Triton2":
        print(
            f"INFO benchmark Baseline Triton2 {label}: inf_sandbox_do_not_read"
        )
        return float("inf")
    x = _make_inputs(*shape)[0]
    try:
        # Ensure model exists before timing.
        _model(provider, label)

        def timed():
            with torch.no_grad():
                return _run_provider(provider, label, x)

        return _bench_ms(timed,
                         warmup=5 if label == "default" else 10,
                         rep=20 if label == "default" else 50)
    except Exception as exc:
        print(f"INFO benchmark {provider} {label}: inf_{type(exc).__name__}")
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
        benchmark.run(print_data=True, show_plots=False, save_path=str(ROOT))


if __name__ == "__main__":
    main()
