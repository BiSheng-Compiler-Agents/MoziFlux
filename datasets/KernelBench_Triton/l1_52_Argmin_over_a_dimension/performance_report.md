# Performance Report

## Cannsim setup

- Target: Ascend950 (`TRITON_ASCEND_ARCH=Ascend910_9589` for compile)
- Baseline trace: `/tmp/cannsim_local/l1_52_argmin_baseline/cannsim_20260625025558_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l1_52_argmin_optimized/cannsim_20260625025744_test_kernel/report/trace_core0.json`
- Trace methodology: grid=1 sub-kernel. Baseline computes 1 output for one 1024-wide row tile; optimized computes 64 output columns for one `[64,64]` middle-dimension tile.

## Cannsim trace summary

| Kernel | Outputs/program | wall_cycles | cycles/output | x_events | i_events | Hardware time estimate |
|---|---:|---:|---:|---:|---:|---:|
| Baseline row-streaming | 1 | 3,633 | 3,633.0 | 517 | 19 | 1.453 µs |
| Optimized dim=1 tiled | 64 | 4,127 | 64.5 | 1,436 | 32 | 1.651 µs/program |

Normalized sub-kernel throughput improves by about `3633 / (4127 / 64) = 56.3x` cycles per output.

## Pipeline table

| Kernel | Bottleneck pipe | busy_cyc | Other major pipes | Top critical instruction |
|---|---:|---:|---|---|
| Baseline | SCALARLDST | 1,349 | MTE2 977; VEC 952; PUSHQ 648; SCALAR 628 | `ST_XD_XN_IMM` total 2,258 cycles; `MOV_SRC_TO_DST_ALIGNv2` 965 cycles |
| Optimized | MTE3 | 2,215 | SCALARLDST 1,830; PUSHQ 1,379; MTE2 1,019; VEC 1,016 | `WAIT_FLAG_VEC` on MTE3 1,869 cycles; `ST_XD_XN_IMM` total 11,410 cycles |

## Interpretation

The optimized program does more per-program vector work and stores 64 indices, so raw wall cycles/program are not directly comparable to the one-output baseline program. Per produced output, the optimized middle-dimension path is substantially cheaper and also removes the full `movedim(...).contiguous()` copy from the target host path.

## Hardware latency

`remote_verify` passed correctness and benchmark on physical Ascend NPU.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Baseline1 / Optimized | PyTorch / Optimized |
|---|---:|---:|---:|---:|---:|---:|
| small_dim1 | 0.003991 | 0.018866 | 0.009192 | 0.005203 | 3.532x | 0.774x |
| irregular_dim1 | 0.007642 | 0.062889 | 0.019564 | 0.008374 | 7.472x | 0.912x |
| dim0_path | 0.003663 | 0.102713 | 0.016833 | 0.005708 | 17.819x | 0.638x |
| dim2_path | 0.003683 | 0.030447 | 0.008147 | 0.015030 | 2.018x | 0.244x |
| exact_target | 5.440865 | inf (grid guard) | inf (grid guard) | 11.328570 | n/a | 0.483x |

- Unit test: `UNIT_TEST PASS` for optimized small/irregular dim1, dim0, dim2, exact target, and 2D fallback.
- Geomean over finite Baseline Triton1 comparisons: `5.550x` faster than baseline.
- Exact target: optimized is valid and avoids baseline `coreDim` overflow, but PyTorch/ACL is faster (`5.440865 ms` vs `11.328570 ms`).
