import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import triton

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "33_Gemm_Scale_BatchNorm.py"
BASE_FILE = ROOT / "base_33_Gemm_Scale_BatchNorm.py"
OPT_FILE = ROOT / "opt_33_Gemm_Scale_BatchNorm.py"

_BENCH_SHAPES = [
    ("small_K1024", 64, 1024, 1024),
    ("medium_K4096", 256, 4096, 4096),
    ("required_K8192", 1024, 8192, 8192),
]
_PROVIDER_FILES = {
    "baseline1": INPUT_FILE,
    "baseline2": BASE_FILE,
    "optimized": OPT_FILE,
}
_MODELS = {}
_MODULES = {}


def _load(path: Path, key: str):
    if key in _MODULES:
        return _MODULES[key]
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULES[key] = mod
    return mod


class TorchRef(nn.Module):

    def __init__(self,
                 in_features=1024,
                 out_features=512,
                 scale_shape=None,
                 eps=1e-5,
                 momentum=0.1,
                 device="npu",
                 dtype=torch.float32):
        super().__init__()
        if scale_shape is None:
            scale_shape = (out_features, )
        self.gemm = nn.Linear(in_features,
                              out_features,
                              device=device,
                              dtype=dtype)
        self.scale = nn.Parameter(
            torch.randn(scale_shape, device=device, dtype=dtype))
        self.bn = nn.BatchNorm1d(out_features,
                                 eps=eps,
                                 momentum=momentum,
                                 device=device,
                                 dtype=dtype)

    def forward(self, x):
        return self.bn(self.gemm(x) * self.scale)


def _device():
    return "npu" if hasattr(torch,
                            "npu") and torch.npu.is_available() else "cpu"


def _sync():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def _init_args(in_features, out_features):
    return [in_features, out_features, (out_features, )]


def _model(key, in_features, out_features):
    cache_key = (key, in_features, out_features)
    if cache_key in _MODELS:
        return _MODELS[cache_key]
    dev = _device()
    torch.manual_seed(0)
    if key == "torch_ref":
        model = TorchRef(*_init_args(in_features, out_features),
                         device=dev,
                         dtype=torch.float32)
    else:
        mod = _load(_PROVIDER_FILES[key], key)
        init = _init_args(in_features, out_features)
        model = mod.ModelNew(*init, device=dev, dtype=torch.float32)
    model.train()
    _MODELS[cache_key] = model
    return model


def _make_inputs(batch, in_features):
    torch.manual_seed(123)
    return (torch.rand((batch, in_features),
                       device=_device(),
                       dtype=torch.float32), )


def _run_torch_ref(batch, in_features, out_features):
    x, = _make_inputs(batch, in_features)
    return _model("torch_ref", in_features, out_features)(x)


def _run_provider(key, batch, in_features, out_features):
    x, = _make_inputs(batch, in_features)
    return _model(key, in_features, out_features)(x)


def _assert_close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)


def unit_test():
    ok = True
    for label, batch, in_features, out_features in _BENCH_SHAPES:
        try:
            ref = _run_torch_ref(batch, in_features, out_features)
            _sync()
        except Exception as e:
            print(
                f"UNIT_TEST_FAILED torch_ref {label}: {type(e).__name__}: {e}")
            ok = False
            continue
        for key, name in [("baseline1", "Baseline Triton1"),
                          ("baseline2", "Baseline Triton2"),
                          ("optimized", "Optimized Triton")]:
            try:
                got = _run_provider(key, batch, in_features, out_features)
                _sync()
                _assert_close(got, ref)
                print(f"UNIT_TEST PASS {name} {label}")
            except Exception as e:
                if key == "optimized":
                    print(
                        f"UNIT_TEST_FAILED {name} {label}: {type(e).__name__}: {e}"
                    )
                    ok = False
                else:
                    print(
                        f"INFO comparison_provider_unavailable {name} {label}: {type(e).__name__}"
                    )
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_one(fn, warmup=25, rep=100):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        # Fallback for Triton-Ascend builds where do_bench is unavailable/unreliable.
        for _ in range(10):
            fn()
        _sync()
        import time
        t0 = time.perf_counter()
        for _ in range(50):
            fn()
        _sync()
        return (time.perf_counter() - t0) * 1000.0 / 50.0


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
        styles=[("blue", "-"), ("red", "--"), ("black", "--"), ("green", "-")],
        ylabel="latency (ms)",
        plot_name="gemm_scale_batchnorm_perf",
        args={},
    ))
def bench(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, batch, in_features, out_features = shape
    if provider == "torch_ref":

        def fn():
            return _run_torch_ref(batch, in_features, out_features)
    else:

        def fn():
            return _run_provider(provider, batch, in_features, out_features)

    try:
        # First run catches provider-wide MLIR/runtime issues before timing loops.
        fn()
        _sync()
        return _bench_one(fn)
    except Exception as e:
        print(f"INFO {provider} {label} benchmark_inf: {type(e).__name__}")
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
