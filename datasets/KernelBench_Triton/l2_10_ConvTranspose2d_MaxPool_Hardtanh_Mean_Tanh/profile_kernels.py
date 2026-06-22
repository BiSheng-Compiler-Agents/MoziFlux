import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import triton

ROOT = Path(__file__).resolve().parent
INPUT = "10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh.py"
BASE2 = "base_10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh.py"
OPT = "opt_10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh.py"


def _load(fname, key):
    path = ROOT / fname
    spec = importlib.util.spec_from_file_location(f"k_{key}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_optional(fname, key):
    try:
        return _load(fname, key)
    except Exception as e:
        print(f"INFO {key} import_unavailable {type(e).__name__}")
        return None


baseline1 = _load(INPUT, "baseline1")
baseline2 = _load_optional(BASE2, "baseline2")
optmod = _load(OPT, "optimized")

_PROVIDERS = ["torch", "baseline1", "baseline2", "optimized"]
_MODEL_CACHE = {}
_BENCH_SHAPES = [
    ("direct_small", 2, 64, 64, 16, 16),
    ("direct_medium", 8, 64, 64, 64, 64),
    ("target_persistent", 128, 64, 64, 256, 256),
]


def _init_args():
    init = baseline1.get_init_inputs() if hasattr(baseline1,
                                                  "get_init_inputs") else []
    if init == [()]:
        init = []
    return list(init)


def _make_inputs(B, in_c, out_c, H, W):
    # Match source: torch.rand default fp32.
    return torch.rand(B, in_c, H, W, device="npu")


def _torch_model():
    key = ("torch", )
    if key not in _MODEL_CACHE:
        torch.manual_seed(0)
        args = _init_args()
        m = nn.Sequential()
        # Manual reference with exactly the same construction order as ModelNew.
        conv = nn.ConvTranspose2d(args[0],
                                  args[1],
                                  args[2],
                                  stride=args[3],
                                  padding=args[4])
        maxpool = nn.MaxPool2d(kernel_size=args[5], stride=args[6])
        hardtanh = nn.Hardtanh(min_val=args[7], max_val=args[8])
        m.conv = conv
        m.maxpool = maxpool
        m.hardtanh = hardtanh
        m.to(device="npu").eval()
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


def _model(provider):
    key = (provider, )
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    mod = {
        "baseline1": baseline1,
        "baseline2": baseline2,
        "optimized": optmod
    }[provider]
    if mod is None:
        return None
    torch.manual_seed(0)
    m = mod.ModelNew(*_init_args()).to(device="npu").eval()
    _MODEL_CACHE[key] = m
    return m


def _run_torch_ref(x):
    m = _torch_model()
    y = m.conv(x)
    y = m.maxpool(y)
    y = m.hardtanh(y)
    y = y.mean(dim=(2, 3), keepdim=True)
    return torch.tanh(y)


def _unavailable(provider, label):
    if provider == "torch" or provider == "optimized":
        return None
    if provider == "baseline2":
        return "read_only_optional_provider_preskip"
    if provider == "baseline1" and label == "target_persistent":
        return "baseline_huge_block_preskip"
    return None


def _run_provider(provider, x):
    if provider == "torch":
        return _run_torch_ref(x)
    m = _model(provider)
    if m is None:
        raise RuntimeError("provider_unavailable")
    return m(x)


def _sync():
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _close(a, b):
    if a.shape != b.shape:
        return False, float("inf"), float("inf")
    diff = (a.float() - b.float()).abs()
    mx = float(diff.max().detach().cpu())
    ref = b.float().abs().clamp_min(1e-6)
    rel = float((diff / ref).max().detach().cpu())
    return bool(torch.allclose(a, b, rtol=1e-3, atol=1e-3)), mx, rel


def unit_test():
    ok_all = True
    with torch.no_grad():
        for label, B, in_c, out_c, H, W in _BENCH_SHAPES:
            x = _make_inputs(B, in_c, out_c, H, W)
            ref = _run_torch_ref(x)
            _sync()
            for provider in ["baseline1", "baseline2", "optimized"]:
                reason = _unavailable(provider, label)
                if reason:
                    print(f"TEST {provider} {label} SKIP {reason}")
                    continue
                try:
                    y = _run_provider(provider, x)
                    _sync()
                    ok, max_abs, max_rel = _close(y, ref)
                    if ok:
                        print(
                            f"TEST {provider} {label} PASS max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
                        )
                    else:
                        print(
                            f"TEST {provider} {label} SKIP value_mismatch max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
                        )
                        if provider == "optimized":
                            ok_all = False
                except Exception as e:
                    tag = "optimized_exception" if provider == "optimized" else "provider_unavailable"
                    print(
                        f"TEST {provider} {label} SKIP {tag} {type(e).__name__}"
                    )
                    if provider == "optimized":
                        ok_all = False
            del x, ref
            _sync()
    print("UNIT_TEST PASS" if ok_all else "UNIT_TEST_FAILED")
    return ok_all


def _bench_callable(fn):
    for _ in range(2):
        out = fn()
        _sync()
        del out
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        out = fn()
        _sync()
        del out
    return (time.perf_counter() - t0) * 1000.0 / reps


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="provider",
        line_vals=_PROVIDERS,
        line_names=[
            "PyTorch / ACL", "Baseline Triton1", "Baseline Triton2",
            "Optimized Triton"
        ],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "-")],
        ylabel="ms",
        plot_name="convtranspose_maxpool_hardtanh_mean_tanh",
        args={},
    ))
def benchmark(label, provider):
    shape = next(s for s in _BENCH_SHAPES if s[0] == label)
    _, B, in_c, out_c, H, W = shape
    reason = _unavailable(provider, label)
    if reason:
        print(f"INFO {provider} {label} INF {reason}")
        return float("inf")
    x = _make_inputs(B, in_c, out_c, H, W)

    def fn():
        return _run_provider(provider, x)

    try:
        return _bench_callable(fn)
    except Exception as e:
        print(f"INFO {provider} {label} INF {type(e).__name__}")
        return float("inf")


def main():
    unit_test()
    benchmark.run(print_data=True, show_plots=False, save_path=None)


if __name__ == "__main__":
    main()
