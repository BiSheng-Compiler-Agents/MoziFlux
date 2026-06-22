# Performance Report

## Cannsim setup

- Baseline probe: `cannsim_baseline`, grid=(1,), `n=4096`, kernel `_smooth_l1_mean_atomic_kernel`.
- Optimized probe: `cannsim_optimized`, grid=(1,), `n=4096`, kernel `_huber_stage1_kernel` from the two-phase diagnostic fallback.
- Trace paths:
  - Baseline: `/tmp/cannsim_local/kernelbench_l1_96_huber_baseline/cannsim_20260625074103_test_kernel/report/trace_core0.json`
  - Optimized: `/tmp/cannsim_local/kernelbench_l1_96_huber_optimized/cannsim_20260625074304_test_kernel/report/trace_core0.json`

## Cannsim trace comparison

| Metric | Baseline atomic mean | Optimized stage1 fallback | Change |
|---|---:|---:|---:|
| wall_cycles | 7,573 | 4,732 | -37.5% |
| x_events | 1,469 | 1,109 | -24.5% |
| i_events | 91 | 26 | -71.4% |
| simulated hardware time (`cycles * 0.4 ns`) | 3.029 us | 1.893 us | -37.5% |

## Pipeline utilization

| Pipeline | Baseline busy_cyc | Baseline ops | Optimized busy_cyc | Optimized ops | Notes |
|---|---:|---:|---:|---:|---|
| MTE2 | 3,013 | 22 | 1,071 | 7 | fewer GM/UB movement events in fallback stage1 |
| VEC | 2,971 | 8 | 1,060 | 2 | fewer waits around vector loop |
| PUSHQ | 2,289 | 30 | 1,093 | 7 | fewer dispatch/control pushes |
| SCALARLDST | 1,967 | 36 | 1,823 | 13 | still the optimized sub-kernel bottleneck |
| SCALAR | 1,718 | 412 | 649 | 166 | scalar/control reduced by private partial storage |
| RVECEX | 756 | 610 | 543 | 584 | same Huber math, modest improvement |
| MTE3 | 285 | 2 | 735 | 2 | optimized probe stores a partial sum instead of final mean |

## Top critical instructions

| Kernel | Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---:|---:|---:|---:|
| Baseline | MOV_SRC_TO_DST_ALIGNv2 | MTE2 | 8 | 5,870 | 734 |
| Baseline | ST_XD_XN_IMM | SCALARLDST | 14 | 3,927 | 280 |
| Baseline | WAIT_FLAG_MTE2 | VEC | 4 | 2,959 | 740 |
| Optimized | ST_XD_XN_IMM | SCALARLDST | 5 | 2,549 | 510 |
| Optimized | MOV_SRC_TO_DST_ALIGNv2 | MTE2 | 2 | 2,083 | 1,042 |
| Optimized | RV_VLDI | RVECLD | 195 | 1,804 | 9 |

## Hardware latency

`remote_verify` passed correctness and benchmark on physical Ascend NPU.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| small_single | 0.008165 | 0.02510 | 0.039830 | 0.009871 |
| direct_1m | 0.007595 | 0.02183 | 0.054041 | 0.008988 |
| persistent_1g | 5.287467 | inf (grid_guard) | 6.294936 | 5.312604 |

The optimized production path is the ACL SmoothL1 dispatch, so it remains grid-legal at the 1G-element target where the editable baseline is pre-skipped for `ceil(n/4096) > 65535`.
