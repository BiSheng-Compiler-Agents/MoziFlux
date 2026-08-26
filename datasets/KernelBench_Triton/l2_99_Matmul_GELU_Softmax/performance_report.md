# Performance Report

## Correctness and hardware benchmark

Remote hardware verification passed: `UNIT_TEST PASS` for optimized dispatch on all benchmark shapes.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs PyTorch / ACL | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|---:|
| small_16x512x512 | 0.082914 | 0.346480 | 0.335544 | 0.069608 | 1.191x | 4.978x |
| medium_128x2048x2048 | 0.269145 | inf (preskipped) | inf (preskipped) | 0.258522 | 1.041x | inf |
| target_1024x8192x8192 | 7.253624 | inf (preskipped) | inf (preskipped) | 7.247609 | 1.001x | inf |

Comparison baselines were preskipped for medium/target because the original row-wise vector GEMM is prohibitively slow at those GEMM sizes; optimized correctness was still tested on every shape.

## cannsim sub-kernel setup

- Baseline trace: `/tmp/cannsim_local/l2_99_matmul_gelu_softmax_baseline2/cannsim_20260629163308_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l2_99_matmul_gelu_softmax_optimized/cannsim_20260629163459_test_kernel/report/trace_core0.json`
- Sub-kernel: grid=1, B=1, N=128. Baseline uses the original single-tile row path; optimized uses the Triton GELU+softmax epilogue.

## cannsim trace summary

| Kernel | Wall cycles | Hardware latency (cycles x 0.4 ns) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline row-wise GEMM+GELU+Softmax | 4310 | 1724.0 ns | 1998 | 35 | MTE3, 1818 busy cycles |
| Optimized Triton GELU+Softmax epilogue | 3508 | 1403.2 ns | 370 | 17 | SCALARLDST, 1710 busy cycles |

Sub-kernel cycle speedup: `4310 / 3508 = 1.229x`.

## Pipeline utilization

| Kernel | MTE3 busy | SCALARLDST busy | PUSHQ busy | MTE2 busy | VEC busy | RVECEX busy | SCALAR busy | RVECLD busy | RVECST busy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1818 | 1778 | 1700 | 1071 | 1013 | 954 | 751 | 709 | 600 |
| Optimized | 1069 | 1710 | 1083 | 697 | 695 | 228 | 671 | 37 | 50 |

Key reductions from the epilogue-only Triton path: RVECEX 954 -> 228, RVECLD 709 -> 37, RVECST 600 -> 50, and x_events 1998 -> 370.

## Top critical instructions

| Kernel | Critical instructions |
|---|---|
| Baseline | `LDP_XI_XJ_XN` 3021 total cycles; `WAIT_FLAG_VEC@MTE3` 1490-cycle event; `MOV_SRC_TO_DST_ALIGNv2` 2681 total cycles |
| Optimized | `LDP_XI_XJ_XN` 1912 total cycles; `LD_XD_XN_IMM` 1697 total cycles; `WAIT_FLAG_VEC@MTE3` 746-cycle event |
