# Performance Report

## Cannsim setup
- Baseline trace: `/tmp/cannsim_local/kb47_sum_baseline3/cannsim_20260625013728_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/kb47_sum_opt1/cannsim_20260625013952_test_kernel/report/trace_core0.json`
- Sub-kernel: `B=1, M=128, N=128`, `dim=1`, one logical output tile.
- Cycle conversion: `hardware_time_ns = cycles × 0.4`.

## Cannsim summary
| Kernel | wall cycles | Est. HW time (µs) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline dim=1 | 14,715 | 5.886 | 3,804 | 186 | MTE2 |
| Optimized dim=1 | 6,561 | 2.624 | 1,234 | 25 | MTE3 / PUSHQ |

Sub-kernel speedup: `14715 / 6561 = 2.24×` fewer wall cycles.

## Pipeline table
| Kernel | MTE2 busy | VEC busy | SCALAR busy | SCALARLDST busy | PUSHQ busy | MTE3 busy | RVECLD busy | RVECEX busy | RVECST busy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 12,323 | 10,904 | 2,783 | 2,440 | 2,106 | 1,073 | 400 | 680 | 298 |
| Optimized | 1,276 | 1,263 | 1,467 | 1,739 | 3,457 | 4,715 | 3,065 | 1,038 | 1,290 |

## Top instruction changes
| Kernel | Critical instruction | Count | Total cycles | Avg cycles |
|---|---|---:|---:|---:|
| Baseline | `MOV_SRC_TO_DST_ALIGNv2` (MTE2) | 128 | 82,777 | 647 |
| Baseline | `MOV_SPR_XN` (MTE2) | 129 | 72,919 | 565 |
| Optimized | `RV_VLDI` (RVECLD) | 512 | 7,537 | 15 |
| Optimized | `WAIT_FLAG_VEC` (MTE3) | 1 | 4,371 | 4,371 |

## Hardware latency
Remote verification directory: `/home/s00929845/kernel_verify/l1_47_Sum_reduction_over_a_dimension_1782351769`.

Correctness: `UNIT_TEST PASS`. Optimized passed all dispatch paths (`dim0`, `dim1`, `dim2`). Baseline comparison providers reported `UnsupportedLanguageConstruct` for `dim0_small`, so those cells are `inf`; optimized still passed that dispatch path.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| dim1_target | 6.676604 | 21.039068 | 10.482754 | 8.159059 | 2.58× |
| dim1_irregular | 0.005149 | 0.022781 | 0.011199 | 0.011612 | 1.96× |
| dim0_small | 0.003826 | inf | inf | 0.005197 | n/a |
| dim2_small | 0.004948 | 0.006401 | 0.006352 | 0.004992 | 1.28× |

Target-shape result: optimized Triton is 2.58× faster than the editable baseline kernel and 1.28× faster than `base_*.py`; PyTorch / ACL remains faster on this shape (optimized/PyTorch latency ratio 1.22×).
