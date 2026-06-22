# Performance Report

## Cannsim setup

- Tool: `cannsim_local_run(..., gen_report=True)` on Ascend950.
- Scope: post-convolution fused Triton epilogue (`min(D) + softmax(C)`), grid=1 sub-kernel.
- Baseline trace: `/tmp/cannsim_local/l2_24_conv3d_min_softmax_baseline_exact/cannsim_20260625122511_test_kernel/report/trace_core0.json`.
- Optimized Triton trace: `/tmp/cannsim_local/l2_24_conv3d_min_softmax_opt_w32/cannsim_20260625122755_test_kernel/report/trace_core0.json`.
- Target production path uses ACL for the large post-op; the trace below documents the retained fused Triton path used for small/medium shapes.

## Cannsim trace summary

| Metric | Baseline fused | Optimized fused | Delta |
|---|---:|---:|---:|
| wall_cycles | 13,876 | 12,735 | -8.22% |
| hardware time (cycles × 0.4 ns) | 5.550 µs | 5.094 µs | -0.456 µs |
| x_events | 12,364 | 12,364 | 0.00% |
| i_events | 283 | 283 | 0.00% |
| bottleneck | MTE3 | MTE3 | unchanged |

## Pipeline utilization

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Delta |
|---|---:|---:|---:|
| MTE3 | 10,005 | 10,067 | +0.62% |
| PUSHQ | 9,741 | 9,743 | +0.02% |
| RVECLD | 6,226 | 6,226 | 0.00% |
| RVECEX | 5,790 | 5,790 | 0.00% |
| MTE2 | 4,453 | 4,524 | +1.59% |
| VEC | 2,722 | 2,792 | +2.57% |
| SCALAR | 2,539 | 1,335 | -47.42% |
| RVECST | 1,953 | 1,953 | 0.00% |
| SCALARLDST | 1,815 | 1,827 | +0.66% |

## Hardware verification

`remote_verify` passed correctness and benchmark.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small | 0.100837 | 0.013226 | 0.031650 | 0.013277 | 0.996x |
| medium | 0.139859 | 0.088327 | 0.164544 | 0.088387 | 0.999x |
| target | 1.547628 | 1.771558 | 3.134761 | 1.439981 | 1.230x |

Geomean speedup vs Baseline Triton1: `1.070x`.

## Correctness

All optimized tests passed:

- `small max_abs=2.235174e-08`
- `medium max_abs=2.980232e-08`
- `target max_abs=0.000000e+00`
- `fallback_c80 max_abs=0.000000e+00`
