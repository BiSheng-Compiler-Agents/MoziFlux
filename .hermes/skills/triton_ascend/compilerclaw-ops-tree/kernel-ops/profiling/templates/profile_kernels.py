"""
profile_kernels.py -- canonical Ascend NPU profiler template

Compares:
  torch_ref  : PyTorch / ACL reference path
  baseline1  : original Triton kernel (<N>_<KernelName>.py)
  baseline2  : base_<N>_<KernelName>.py if present; otherwise visible SKIP rows
  optimized  : optimized Triton kernel (opt_<N>_<KernelName>.py)

This template is intentionally parser-friendly for kernel-sandbox
_validate_results_txt:
  - prints UNIT_TEST PASS / UNIT_TEST_FAILED
  - prints canonical TEST lines containing provider keys: baseline1, baseline2, optimized
  - prints a benchmark table with PyTorch / ACL, Baseline Triton1/2, Optimized Triton

Benchmarking uses torch_npu.profiler and parses ASCEND_PROFILER_OUTPUT/op_statistic.csv
Avg Time(us). If op_statistic is unavailable, it falls back to kernel_details.csv.

Usage:
    python profile_kernels.py            # correctness + benchmark
    python profile_kernels.py --test     # correctness only
    python profile_kernels.py --bench    # benchmark only
"""
import argparse
import csv
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
from torch_npu.profiler import (
    ExportType,
    ProfilerActivity,
    ProfilerLevel,
    _ExperimentalConfig,
    profile,
    schedule,
    tensorboard_trace_handler,
)

_DIR = Path(__file__).parent

# ---- Adapt these filenames per kernel workspace ---------------------------
BASELINE1_FILE = "19_ReLU.py"
BASELINE2_FILE = "base_19_ReLU.py"  # keep visible even when absent
OPTIMIZED_FILE = "opt_19_ReLU.py"

# Optional filters for profiler CSV rows. Leave None to sum all op_statistic rows
# emitted by one provider call. For ACL fused attention, use e.g. "FlashAttention".
_PROVIDER_OP_FILTERS = {
    "torch_ref": None,
    "baseline1": None,
    "baseline2": None,
    "optimized": None,
}


def _load_optional(fname):
    path = _DIR / fname
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("k_" + path.stem, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        return exc


_baseline_mod1 = _load_optional(BASELINE1_FILE)
_baseline_mod2 = _load_optional(BASELINE2_FILE)
_optimized_mod = _load_optional(OPTIMIZED_FILE)

# torch_ref first, then comparison/optimized variants. Provider keys must remain
# lowercase baseline1/baseline2/optimized for kernel-sandbox validation.
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
    "optimized": _optimized_mod,
}

# ---- Lazy NPU model init ---------------------------------------------------
_models = {}


def _model(key):
    mod = _MODS.get(key)
    if mod is None:
        raise RuntimeError(f"provider module missing: {key}")
    if isinstance(mod, Exception):
        raise RuntimeError(
            f"provider module import failed: {key}: {type(mod).__name__}")
    if key not in _models:
        torch.manual_seed(0)
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        if init == [()]:
            init = []
        _models[key] = mod.ModelNew(*init).to(device="npu").eval()
    return _models[key]


# ---- Adapt these runners per kernel ---------------------------------------
def _run_torch_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation. Replace for each kernel."""
    return torch.nn.functional.relu(x)


def _run_provider(key, x: torch.Tensor) -> torch.Tensor:
    if key == "torch_ref":
        return _run_torch_ref(x)
    return _model(key)(x)


# Labels must contain no spaces. _BENCH_SHAPES is the single source of truth
# for both unit test and benchmark.
_BENCH_SHAPES = [
    # (label, n_elements)
    ("N1024", 1024),
    ("N65536", 65536),
    ("N524288", 524288),
]


def _make_inputs(label):
    n = next(s[1] for s in _BENCH_SHAPES if s[0] == label)
    torch.manual_seed(42)
    x = torch.rand(n, device="npu", dtype=torch.float16) * 4 - 2
    return (x, )


def _sync():
    torch.npu.synchronize()


# ---- Correctness: canonical TEST lines for kernel-sandbox ------------------
def unit_test():
    print("=== Unit Test: every variant vs torch_ref ===")
    optimized_fail = False
    reference_fail = False

    for label, *_ in _BENCH_SHAPES:
        inputs = _make_inputs(label)
        try:
            with torch.no_grad():
                ref = _run_torch_ref(*inputs)
                _sync()
        except Exception as exc:
            reference_fail = True
            print(
                f"TEST baseline1 {label}: SKIP_REFERENCE_ERROR {type(exc).__name__} max_abs=inf"
            )
            print(
                f"TEST baseline2 {label}: SKIP_REFERENCE_ERROR {type(exc).__name__} max_abs=inf"
            )
            print(
                f"TEST optimized {label}: FAIL_REFERENCE_ERROR {type(exc).__name__} max_abs=inf"
            )
            optimized_fail = True
            continue

        for key, _display in _VARIANTS:
            try:
                with torch.no_grad():
                    out = _run_provider(key, *inputs)
                    _sync()
                max_abs = (ref.float() - out.float()).abs().max().item()
                mean_abs = (ref.float() - out.float()).abs().mean().item()
                ok = bool(
                    torch.allclose(ref.float(),
                                   out.float(),
                                   atol=1e-2,
                                   rtol=1e-2))
                status = "PASS" if ok else "FAIL"
                print(f"TEST {key} {label}: {status} "
                      f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}")
                if key == "optimized" and not ok:
                    optimized_fail = True
            except Exception as exc:
                # Keep comparison provider entries visible, but only optimized failure gates UNIT_TEST.
                status = "FAIL" if key == "optimized" else "SKIP_UNAVAILABLE"
                print(
                    f"TEST {key} {label}: {status} {type(exc).__name__} max_abs=inf mean_abs=inf"
                )
                if key == "optimized":
                    optimized_fail = True

    if optimized_fail or reference_fail:
        print("UNIT_TEST_FAILED")
        return False
    print("UNIT_TEST PASS")
    return True


# ---- torch_npu.profiler benchmark -----------------------------------------
def _profiler_op_ms(fn,
                    label,
                    mode,
                    op_type_filter=None,
                    wait=1,
                    warmup=2,
                    active=5):
    """Return device average runtime in ms from op_statistic.csv.

    This profiles exactly `fn()` inside the profiler context, synchronizing before
    the loop and after each iteration for clean step attribution.
    """
    out_dir = tempfile.mkdtemp(prefix=f"prof_{mode}_{label}_")
    try:
        cfg = _ExperimentalConfig(
            profiler_level=ProfilerLevel.
            Level0,  # minimal overhead, timing only
            export_type=[ExportType.Text],
        )
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                schedule=schedule(wait=wait,
                                  warmup=warmup,
                                  active=active,
                                  repeat=1),
                on_trace_ready=tensorboard_trace_handler(out_dir),
                record_shapes=True,
                experimental_config=cfg,
        ) as prof:
            _sync()
            for _ in range(wait + warmup + active):
                fn()
                _sync()
                prof.step()
        _sync()

        # Preferred: op_statistic.csv Avg Time(us). It is already averaged over active steps.
        op_avgs_us = []
        for root, _, files in os.walk(out_dir):
            if "op_statistic.csv" in files:
                with open(os.path.join(root, "op_statistic.csv"),
                          newline="") as f:
                    for row in csv.DictReader(f):
                        op_type = row.get("OP Type", "")
                        if op_type_filter and op_type_filter not in op_type:
                            continue
                        try:
                            op_avgs_us.append(float(row["Avg Time(us)"]))
                        except Exception:
                            pass
        if op_avgs_us:
            return sum(op_avgs_us) / 1000.0

        # Fallback: average matching rows in kernel_details.csv.
        durations_us = []
        for root, _, files in os.walk(out_dir):
            if "kernel_details.csv" in files:
                with open(os.path.join(root, "kernel_details.csv"),
                          newline="") as f:
                    for row in csv.DictReader(f):
                        name = row.get("Name", "")
                        ktype = row.get("Type", "")
                        if op_type_filter and op_type_filter not in name and op_type_filter not in ktype:
                            continue
                        try:
                            durations_us.append(float(row["Duration(us)"]))
                        except Exception:
                            pass
        if durations_us:
            return (sum(durations_us) / len(durations_us)) / 1000.0

        print(f"INFO profiler_no_matching_rows {mode} {label} dir={out_dir}")
        return float("inf")
    finally:
        if os.environ.get("KEEP_PROFILE_DIR", "0") != "1":
            shutil.rmtree(out_dir, ignore_errors=True)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=[k for k, _ in _PROVIDERS],
        line_names=[n for _, n in _PROVIDERS],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "--")],
        ylabel="Latency (ms)",
        plot_name="kernel_perf",
        args={},
    ))
def benchmark(label, mode):
    inputs = _make_inputs(label)
    try:
        with torch.no_grad():
            return _profiler_op_ms(
                lambda: _run_provider(mode, *inputs),
                label,
                mode,
                op_type_filter=_PROVIDER_OP_FILTERS.get(mode),
            )
    except Exception as exc:
        print(f"INFO bench_unavailable {mode} {label}: {type(exc).__name__}")
        return float("inf")


# ---- Entry point -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Profile kernels on Ascend NPU")
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

    # Always exit zero so remote_verify can download results.txt and benchmark artifacts.
    if ok:
        print("PROFILE_RESULT ok")
    else:
        print("PROFILE_RESULT correctness_failed")
    sys.exit(0)


if __name__ == "__main__":
    main()
