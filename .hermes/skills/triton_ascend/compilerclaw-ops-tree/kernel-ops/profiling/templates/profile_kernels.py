"""
profile_kernels.py -- l1_19_ReLU

Compares three implementations on Ascend NPU hardware:
  torch_ref  : torch.nn.functional.relu (ACL built-in path)
  baseline   : original Triton kernel (19_ReLU.py, autotune + 1D grid)
  optimized  : persistent-grid + fp32-upcast kernel (opt_19_ReLU.py)

Correctness checks every variant against torch_ref in isolated try/except
cells. The benchmark is resilient: a failing provider yields inf, not a crash.
The table format is compatible with generate_report.py (x_names=["label"],
space-free labels, recognized method names).

Usage:
    python profile_kernels.py            # correctness + benchmark
    python profile_kernels.py --test     # correctness only
    python profile_kernels.py --bench    # benchmark only
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

_DIR = Path(__file__).parent


def _load(fname):
    spec = importlib.util.spec_from_file_location("k_" + Path(fname).stem,
                                                  _DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_baseline_mod1 = _load("19_ReLU.py")
_baseline_mod2 = _load("base_19_ReLU.py")  # optional (if available)
_optimized_mod = _load("opt_19_ReLU.py")

# ── Provider map ─────────────────────────────────────────────────────────────
# torch_ref first, then Triton variants. _VARIANTS excludes torch_ref.
_PROVIDERS = [
    ("torch_ref", "PyTorch / ACL"),
    ("baseline1", "Baseline Triton1"),
    ("baseline2", "Baseline Triton2"),
    ("optimized", "Optimized Triton"),
]
_VARIANTS = [(k, n) for k, n in _PROVIDERS if k != "torch_ref"]
_MODS = {
    "baseline1": _baseline_mod1,
    "baseline2": _baseline_mod2,
    "optimized": _optimized_mod
}

# ── Lazy NPU model init ──────────────────────────────────────────────────────
# Do NOT instantiate ModelNew at module level — it creates on CPU and breaks
# when called with NPU input. Lazy-init on first use instead.
_models = {}


def _model(key):
    if key not in _models:
        torch.manual_seed(0)
        init = _MODS[key].get_init_inputs() if hasattr(
            _MODS[key], "get_init_inputs") else []
        # get_init_inputs may return [()] meaning "no args"; flatten appropriately
        if init == [()]:
            init = []
        _models[key] = _MODS[key].ModelNew(*init).to(
            device="npu", dtype=torch.float32).eval()
    return _models[key]


# ── Runners ──────────────────────────────────────────────────────────────────
def _run_torch_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch built-in ReLU — routes to Huawei ACL path on Ascend."""
    return torch.nn.functional.relu(x)


def _run_provider(key, x: torch.Tensor) -> torch.Tensor:
    if key == "torch_ref":
        return _run_torch_ref(x)
    return _model(key)(x)


# ── Shapes ───────────────────────────────────────────────────────────────────
# Labels MUST contain NO spaces (generate_report.py parser splits on whitespace).
_BENCH_SHAPES = [
    # (label,             n_elements)
    ("N1024", 1024),
    ("N65536", 65536),
    ("N524288", 524288),
    ("N4M", 4194304),
    ("N16M", 16777216),
    ("Nbench-4096x393216", 4096 * 393216),
]


# ── Correctness: every variant vs torch_ref ─────────────────────────────────
def unit_test():
    print("=== Unit Test: l1_19_ReLU — every variant vs torch_ref ===")
    any_fail = False

    for label, n in _BENCH_SHAPES:
        torch.manual_seed(42)
        x = torch.rand(n, device="npu", dtype=torch.float16) * 4 - 2
        with torch.no_grad():
            ref = _run_torch_ref(x)

        cells = []
        for key, name in _VARIANTS:
            try:
                with torch.no_grad():
                    out = _run_provider(key, x)
                ok = torch.allclose(ref.float(),
                                    out.float(),
                                    atol=1e-2,
                                    rtol=1e-2)
                diff = (ref.float() - out.float()).abs().max().item()
                cells.append(f"{name}=[{'PASS' if ok else 'FAIL'} {diff:.2e}]")
                if not ok:
                    any_fail = True
            except Exception as e:
                cells.append(
                    f"{name}=[ERROR:{type(e).__name__}: {str(e)[:50]}]")
                any_fail = True
        print(f"  [{label:<20}] " + "  ".join(cells))

    print()
    print("FAILED: some shapes did not pass correctness check"
          if any_fail else "All shapes PASSED")
    return not any_fail


# ── Benchmark ────────────────────────────────────────────────────────────────
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=[k for k, _ in _PROVIDERS],
        line_names=[n for _, n in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="relu_perf",
        args={},
    ))
def benchmark(label, mode):
    n = next(s[1] for s in _BENCH_SHAPES if s[0] == label)
    x = torch.rand(n, device="npu", dtype=torch.float16) * 4 - 2
    try:

        def _benchmark():
            return _run_provider(mode, x)

        return triton.testing.do_bench(
            _benchmark,
            warmup=25,
            rep=200,
            return_mode="mean",
        )
    except Exception as e:
        print(f"[BENCH] {mode} failed on {label}: {type(e).__name__}: {e}")
        return float("inf")


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Profile l1_19_ReLU kernels")
    parser.add_argument("--test",
                        action="store_true",
                        help="Correctness check only")
    parser.add_argument("--bench", action="store_true", help="Benchmark only")
    args = parser.parse_args()

    run_test = args.test or not args.bench
    run_bench = args.bench or not args.test

    ok = True
    if run_test:
        ok = unit_test() and ok
    if run_bench:
        benchmark.run(save_path=str(_DIR), print_data=True)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
