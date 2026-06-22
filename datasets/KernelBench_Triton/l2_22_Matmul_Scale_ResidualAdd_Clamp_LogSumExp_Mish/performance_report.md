# Performance Report

## Cannsim setup

Sub-kernel: post-GEMM row reduction/Mish body only, `B=1`, `N=1024`, `BLOCK_N=1024`, `grid=(1,)`, fp32 buffers. Full-shape dispatch effects of the ACL path are measured by hardware profiling, not by this sub-kernel trace.

Trace files:
- Baseline: `/tmp/cannsim_local/l2_22_baseline_post/cannsim_20260625115247_test_kernel/report/trace_core0.json`
- Optimized: `/tmp/cannsim_local/l2_22_optimized_post/cannsim_20260625115459_test_kernel/report/trace_core0.json`

## Cannsim trace comparison

| Metric | Baseline Triton post-op | Optimized Triton fallback | Change |
|---|---:|---:|---:|
| wall_cycles | 10,460 | 9,807 | -6.2% |
| hardware time (`cycles*0.4ns`) | 4.184 us | 3.923 us | -0.261 us |
| x_events | 1,251 | 1,240 | -0.9% |
| i_events | 100 | 101 | +1 |

## Pipeline busy cycles

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Change |
|---|---:|---:|---:|
| PUSHQ | 6,361 | 6,105 | -4.0% |
| SCALARLDST | 3,043 | 2,447 | -19.6% |
| SCALAR | 2,339 | 1,478 | -36.8% |
| MTE2 | 999 | 980 | -1.9% |
| VEC | 990 | 971 | -1.9% |
| RVECEX | 736 | 734 | -0.3% |
| RVECLD | 398 | 407 | +2.3% |
| RVECST | 339 | 377 | +11.2% |
| MTE3 | 136 | 137 | +0.7% |
| FLOWCTRL | 7 | 7 | 0% |

## Top instruction deltas

| Instruction | Baseline | Optimized | Note |
|---|---:|---:|---|
| `VF` total_cyc | 6,182 | 5,971 | PUSHQ remains bottleneck |
| `LDP_XI_XJ_XN` total_cyc | 2,695 | 1,535 | lower scalar load pressure |
| `LD_XD_XN_IMM` total_cyc | 2,577 | 622 | fewer/cheaper scalar loads |
| `WAIT_FLAG_MTE2` total_cyc | 986 | 968 | minor memory-wait reduction |

## Hardware profiling

`remote_verify` passed correctness and benchmark on physical Ascend hardware.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 | Speedup vs PyTorch |
|---|---:|---:|---:|---:|---:|---:|
| small_triton | 0.047480 | 0.020534 | 0.018077 | 0.020070 | 1.023x | 2.366x |
| medium_acl | 0.166260 | 0.147096 | 0.140203 | 0.146245 | 1.006x | 1.137x |
| target_acl | 1.059926 | 1.145896 | 1.053282 | 1.054515 | 1.087x | 1.005x |

Geomean speedup vs Baseline Triton1: 1.038x. The target optimization win is dispatch-level: keep the tuned ACL large reduction path while retaining the improved Triton fallback for small/medium legal row grids.
