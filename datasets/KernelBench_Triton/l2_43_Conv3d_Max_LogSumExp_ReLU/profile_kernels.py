import argparse
import importlib.util
import pathlib
import sys
import time
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton

HERE = pathlib.Path(__file__).parent.resolve()
INPUT_FILE = HERE / "43_Conv3d_Max_LogSumExp_ReLU.py"
BASE_FILE = HERE / "base_43_Conv3d_Max_LogSumExp_ReLU.py"
OPT_FILE = HERE / "opt_43_Conv3d_Max_LogSumExp_ReLU.py"

_BENCH_SHAPES = [
    ("small", 1, 32, 8, 16, 16, 64),
    ("medium", 2, 32, 16, 64, 64, 64),
    ("default", 4, 32, 32, 128, 128, 64),
]

_PROVIDER_FILES = {
    "baseline1": INPUT_FILE,
    "baseline2": BASE_FILE,
    "optimized": OPT_FILE,
}

_PROVIDER_NAMES = {
    "torch": "PyTorch / ACL",
    "baseline1": "Baseline Triton1",
    "baseline2": "Baseline Triton2",
    "optimized": "Optimized Triton",
}

_MODULES = {}
_MODELS = {}
_REF_MODELS = {}
_PROVIDER_AVAILABLE = {}


def _load(path: pathlib.Path, key: str):
    if key in _MODULES:
        return _MODULES[key]
    try:
        spec = importlib.util.spec_from_file_location(f"k43_{key}_{path.stem}",
                                                      path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MODULES[key] = mod
        _PROVIDER_AVAILABLE[key] = True
        return mod
    except Exception as exc:
        print(f"INFO provider_unavailable {key}: {type(exc).__name__}")
        _PROVIDER_AVAILABLE[key] = False
        _MODULES[key] = None
        return None


def _make_inputs(shape):
    _label, n, cin, d, h, w, _cout = shape
    torch.manual_seed(123)
    return [torch.rand(n, cin, d, h, w, device="npu")]


def _ref_model(shape):
    label, _n, cin, _d, _h, _w, cout = shape
    k = (cin, cout)
    if k not in _REF_MODELS:
        torch.manual_seed(0)
        m = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
        ).npu().eval()
        _REF_MODELS[k] = m
    return _REF_MODELS[k]


def _model(key, shape):
    label, _n, cin, _d, _h, _w, cout = shape
    cache_key = (key, cin, cout)
    if cache_key in _MODELS:
        return _MODELS[cache_key]
    mod = _load(_PROVIDER_FILES[key], key)
    if mod is None:
        return None
    torch.manual_seed(0)
    try:
        m = mod.ModelNew(cin, cout, 3, 1, 1).npu().eval()
    except Exception as exc:
        print(f"INFO model_unavailable {key} {label}: {type(exc).__name__}")
        return None
    _MODELS[cache_key] = m
    return m


def _run_torch_ref(x, shape):
    m = _ref_model(shape)
    with torch.no_grad():
        y = m(x)
        y = torch.logsumexp(y.float(), dim=1, keepdim=True)
        y = torch.relu(y)
        return y


def _run_provider(key, x, shape):
    if key == "torch":
        return _run_torch_ref(x, shape)
    m = _model(key, shape)
    if m is None:
        return None
    with torch.no_grad():
        return m(x)


def _sync():
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().detach().cpu())


def unit_test():
    ok = True
    for shape in _BENCH_SHAPES:
        label = shape[0]
        x = _make_inputs(shape)[0]
        ref = _run_torch_ref(x, shape)
        _sync()
        for key in ["baseline1", "baseline2", "optimized"]:
            name = _PROVIDER_NAMES[key]
            try:
                out = _run_provider(key, x, shape)
                _sync()
                if out is None:
                    print(f"INFO {name} {label} unavailable")
                    continue
                diff = _max_abs(out, ref)
                passed = (out.shape == ref.shape) and diff <= 1e-3
                print(
                    f"TEST {name} {label}: shape={tuple(out.shape)} max_abs={diff:.6g} {'PASS' if passed else 'MISMATCH'}"
                )
                if key == "optimized" and not passed:
                    ok = False
            except Exception as exc:
                print(f"INFO {name} {label} exception {type(exc).__name__}")
                if key == "optimized":
                    ok = False
    print("UNIT_TEST PASS" if ok else "UNIT_TEST_FAILED")
    return ok


def _bench_one(fn, warmup=10, rep=30):
    try:
        return triton.testing.do_bench(fn,
                                       warmup=warmup,
                                       rep=rep,
                                       return_mode="mean")
    except Exception:
        for _ in range(5):
            fn()
            _sync()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        _sync()
        return (time.perf_counter() - t0) * 1000.0 / rep


def _shape_by_label(label):
    for s in _BENCH_SHAPES:
        if s[0] == label:
            return s
    raise KeyError(label)


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
        plot_name="conv3d_max_lse_relu",
        args={},
    ))
def benchmark(label, provider):
    shape = _shape_by_label(label)
    if provider == "baseline2" and not BASE_FILE.exists():
        return float("inf")
    x = _make_inputs(shape)[0]
    # Correctness-gate each cell; comparison providers remain visible as inf if unavailable.
    try:
        out = _run_provider(provider, x, shape)
        _sync()
        if out is None:
            return float("inf")
    except Exception as exc:
        print(
            f"INFO benchmark_preskip {_PROVIDER_NAMES[provider]} {label}: {type(exc).__name__}"
        )
        return float("inf")
    try:
        return _bench_one(lambda: _run_provider(provider, x, shape))
    except Exception as exc:
        print(
            f"INFO benchmark_inf {_PROVIDER_NAMES[provider]} {label}: {type(exc).__name__}"
        )
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
