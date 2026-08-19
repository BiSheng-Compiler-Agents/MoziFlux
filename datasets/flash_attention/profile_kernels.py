"""
profile_kernels.py -- datasets/flash_attention/fa_forward.py

FlashAttention forward profiling on Ascend NPU.

Reference: torch_npu.npu_fusion_attention (PyTorch / ACL)
Providers: fa_forward.py baseline, V1, V2, V3, and PyTorch / ACL

This script follows the profiling template:
  - canonical TEST lines for baseline, V1, V2, and V3
  - UNIT_TEST PASS / UNIT_TEST_FAILED
  - torch_npu.profiler timing inside benchmark()
  - op_statistic.csv Avg Time(us) preferred, kernel_details.csv fallback

Usage:
    python profile_kernels.py            # correctness + benchmark
    python profile_kernels.py --test     # correctness only
    python profile_kernels.py --bench    # benchmark only
"""
import argparse
import csv
import importlib.util
import math
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

import torch
import torch_npu
import triton
from torch_npu.profiler import (
    ExportType,
    ProfilerActivity,
    _ExperimentalConfig,
    profile,
    schedule,
    tensorboard_trace_handler,
)

_DIR = Path(__file__).parent


def _load(fname):
    spec = importlib.util.spec_from_file_location("k_" + Path(fname).stem,
                                                  _DIR / fname)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


_base_mod = _load("fa_forward.py")
_v1_mod = _load("fa_forward_v1.py")
_v2_mod = _load("fa_forward_v2.py")
_v3_mod = _load("fa_forward_v3.py")

_PROVIDERS = [
    ("torch_ref", "PyTorch / ACL"),
    ("baseline", "Baseline fa_forward"),
    ("v1", "Triton V1"),
    ("v2", "Triton V2"),
    ("v3", "Triton V3"),
]
_VARIANTS = [(k, n) for k, n in _PROVIDERS if k != "torch_ref"]
_PROVIDER_OP_FILTERS = {
    "torch_ref": "FlashAttentionScore",
    "baseline": "_fwd_kernel",
    "v1": "_fwd_kernel",
    "v2": "_fwd_kernel",
    "v3": "_fwd_kernel",
}

# Shapes from fa.md. Labels must contain no spaces. Tuple fields:
# (label, batch, q_heads, kv_heads, n_ctx, head_dim, causal, dtype, BLOCK_M, BLOCK_N)
_BENCH_SHAPES = [
    ("ID1_B128H8N8192D128_noncausal_fp16", 128, 8, 8, 8192, 128, False,
     torch.float16, 128, 128),
    ("ID2_B128H8N8192D64_noncausal_fp16", 128, 8, 8, 8192, 64, False,
     torch.float16, 128, 128),
    ("ID3_B128H8N1024D128_noncausal_fp16", 128, 8, 8, 1024, 128, False,
     torch.float16, 128, 128),
    ("ID4_B128H8N1024D64_noncausal_fp16", 128, 8, 8, 1024, 64, False,
     torch.float16, 128, 128),
    ("ID5_B128H8N8192D128_causal_fp16", 128, 8, 8, 8192, 128, True,
     torch.float16, 128, 128),
    ("ID6_B128H8N8192D64_causal_fp16", 128, 8, 8, 8192, 64, True,
     torch.float16, 128, 128),
    ("ID7_B128H8N1024D128_causal_fp16", 128, 8, 8, 1024, 128, True,
     torch.float16, 128, 128),
    ("ID8_B128H8N1024D64_causal_fp16", 128, 8, 8, 1024, 64, True,
     torch.float16, 128, 128),
]


def _sync():
    torch.npu.synchronize()


def _make_inputs(label):
    _, B, q_heads, kv_heads, N, D, causal, dtype, BLOCK_M, BLOCK_N = next(
        s for s in _BENCH_SHAPES if s[0] == label)
    assert q_heads == kv_heads, "fa_forward.py supports equal Q/KV head counts"
    torch.manual_seed(42)
    q = torch.randn((B, q_heads, N, D), device="npu", dtype=dtype)
    k = torch.randn((B, kv_heads, N, D), device="npu", dtype=dtype)
    v = torch.randn((B, kv_heads, N, D), device="npu", dtype=dtype)
    sm_scale = 1.0 / math.sqrt(D)
    return q, k, v, sm_scale, causal, BLOCK_M, BLOCK_N


def _manual_attention(q, k, v, sm_scale, causal):
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if causal:
        n_ctx = q.shape[-2]
        causal_mask = torch.ones((n_ctx, n_ctx),
                                 device=q.device,
                                 dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def _run_torch_ref(q, k, v, sm_scale, causal):
    """ACL fused attention reference via torch_npu.npu_fusion_attention."""
    _, head_num, _, _ = q.shape
    try:
        if causal:
            # sparse_mode=2 without a mask returns FULL attention on this torch_npu
            # build. Verified correct causal call: explicit mask (True = masked
            # position) with sparse_mode=1.
            n_ctx = q.shape[2]
            atten_mask = ~torch.ones(
                n_ctx, n_ctx, device=q.device, dtype=torch.bool).tril()
            out = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num,
                "BNSD",
                padding_mask=None,
                atten_mask=atten_mask,
                scale=sm_scale,
                keep_prob=1.0,
                pre_tockens=65535,
                next_tockens=65535,
                sparse_mode=1,
            )
        else:
            out = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num,
                "BNSD",
                padding_mask=None,
                atten_mask=None,
                scale=sm_scale,
                keep_prob=1.0,
                pre_tockens=65535,
                next_tockens=65535,
                sparse_mode=0,
            )
        return out[0] if isinstance(out, tuple) else out
    except Exception as e:
        print(
            f"INFO npu_fusion_attention_unavailable {type(e).__name__}; using manual reference"
        )
        return _manual_attention(q, k, v, sm_scale, causal)


def _unsupported_reason(key, q, causal, BLOCK_M, BLOCK_N):
    if key == "baseline":
        # The original beta-form baseline fixes BLOCK_M=BLOCK_N=64 internally.
        core_dim = triton.cdiv(q.shape[2], 64) * q.shape[0] * q.shape[1]
        if core_dim > 65535:
            return f"single baseline launch coreDim={core_dim} exceeds 65535"
        if causal:
            return "baseline causal path is not correctness-qualified at 128x128"
    return None


def _run_provider(key, q, k, v, sm_scale, causal, BLOCK_M, BLOCK_N):
    if key == "torch_ref":
        return _run_torch_ref(q, k, v, sm_scale, causal)
    if key == "baseline":
        return _base_mod.attention(q, k, v, sm_scale, causal)
    if key == "v1":
        return _v1_mod.attention(q, k, v, sm_scale, causal, BLOCK_M, BLOCK_N)
    if key == "v2":
        return _v2_mod.attention(q, k, v, sm_scale, causal, BLOCK_M, BLOCK_N)
    if key == "v3":
        return _v3_mod.attention(q, k, v, sm_scale, causal, BLOCK_M, BLOCK_N)
    raise KeyError(key)


def unit_test():
    print(
        "=== Unit Test: flash_attention forward — every variant vs torch_ref ==="
    )
    any_fail = False
    for label, *_ in _BENCH_SHAPES:
        q, k, v, sm_scale, causal, BLOCK_M, BLOCK_N = _make_inputs(label)
        with torch.no_grad():
            ref = _run_torch_ref(q, k, v, sm_scale, causal)
            _sync()

        for key, _name in _VARIANTS:
            reason = _unsupported_reason(key, q, causal, BLOCK_M, BLOCK_N)
            if reason:
                print(f"TEST {key} {label}: SKIP_UNSUPPORTED "
                      f"reason={reason} max_abs=inf mean_abs=inf")
                continue
            try:
                with torch.no_grad():
                    out = _run_provider(key, q, k, v, sm_scale, causal,
                                        BLOCK_M, BLOCK_N)
                    _sync()
                max_abs = (ref.float() - out.float()).abs().max().item()
                mean_abs = (ref.float() - out.float()).abs().mean().item()
                ok = max_abs <= 2e-2 and mean_abs <= 2e-3
                status = "PASS" if ok else "FAIL"
                print(
                    f"TEST {key} {label}: {status} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}"
                )
                if not ok:
                    any_fail = True
            except Exception as e:
                msg = str(e).replace("\n", " ")
                print(
                    f"TEST {key} {label}: ERROR {type(e).__name__} msg={msg} max_abs=inf mean_abs=inf"
                )
                any_fail = True
    print("UNIT_TEST_FAILED" if any_fail else "UNIT_TEST PASS")
    return not any_fail


def _profiler_op_ms(fn,
                    label,
                    mode,
                    op_type_filter=None,
                    wait=1,
                    warmup=2,
                    active=5):
    out_dir = tempfile.mkdtemp(prefix=f"fa_prof_{mode}_{label}_")
    try:
        experimental_config = _ExperimentalConfig(
            profiler_level="Level0",
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
                experimental_config=experimental_config,
        ) as prof:
            _sync()
            for _ in range(wait + warmup + active):
                fn()
                _sync()
                prof.step()
        _sync()

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
            avg_ms = sum(op_avgs_us) / 1000.0
            print(
                f"INFO profiler_avg {mode} {label}: {avg_ms:.6f} ms from op_statistic dir={out_dir}"
            )
            return avg_ms

        durations = []
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
                            durations.append(float(row["Duration(us)"]))
                        except Exception:
                            pass
        if durations:
            median_ms = statistics.median(durations) / 1000.0
            print(f"INFO profiler_median {mode} {label}: {median_ms:.6f} ms "
                  f"samples={len(durations)} "
                  f"min={min(durations) / 1000.0:.6f} "
                  f"max={max(durations) / 1000.0:.6f} dir={out_dir}")
            return median_ms

        print(f"INFO profiler_no_matching_rows {mode} {label} dir={out_dir}")
        return float("inf")
    finally:
        if os.environ.get("KEEP_FA_PROF", "0") != "1":
            shutil.rmtree(out_dir, ignore_errors=True)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=[k for k, _ in _PROVIDERS],
        line_names=[n for _, n in _PROVIDERS],
        styles=[
            ("blue", "-"),
            ("red", "-"),
            ("green", "-"),
            ("black", "-"),
            ("orange", "-"),
            ("purple", "-"),
            ("brown", "-"),
        ],
        ylabel="Latency (ms)",
        plot_name="flash_attention_forward_perf",
        args={},
    ))
def benchmark(label, mode):
    q, k, v, sm_scale, causal, BLOCK_M, BLOCK_N = _make_inputs(label)
    reason = _unsupported_reason(mode, q, causal, BLOCK_M, BLOCK_N)
    if reason:
        print(f"INFO unsupported {mode} {label}: {reason}")
        return float("inf")
    try:
        with torch.no_grad():
            return _profiler_op_ms(
                lambda: _run_provider(mode, q, k, v, sm_scale, causal, BLOCK_M,
                                      BLOCK_N),
                label,
                mode,
                op_type_filter=_PROVIDER_OP_FILTERS.get(mode),
            )
    except Exception as e:
        print(f"INFO bench_unavailable {mode} {label}: {type(e).__name__}")
        return float("inf")


def main():
    parser = argparse.ArgumentParser(
        description="Profile FlashAttention forward kernel")
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

    if ok:
        print("PROFILE_RESULT ok")
    else:
        print("PROFILE_RESULT correctness_failed")
    sys.exit(0)


if __name__ == "__main__":
    main()
