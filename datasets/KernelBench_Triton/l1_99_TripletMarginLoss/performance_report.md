# Performance Report: TripletMarginLoss

## Cannsim setup

- Baseline trace: `_triplet_margin_row_kernel`, one row, `D=8192`, `BLOCK_SIZE=1024`, `N_ITERS=8`, `grid=1`.
- Optimized trace: `_triplet_margin_row_atomic_kernel`, same row shape/blocking, scalar output pre-zeroed, `grid=1`.
- Cycle-to-time conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim trace summary

| Kernel | wall_cycles | est_hw_ns | x_events | i_events | Bottleneck | MTE2 busy | VEC busy | PUSHQ busy | SCALAR busy | SCALARLDST busy |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Baseline row | 13,896 | 5,558.4 | 2,819 | 164 | MTE2 | 6,226 | 5,605 | 4,935 | 2,961 | 2,586 |
| Optimized atomic row | 13,643 | 5,457.2 | 2,844 | 168 | MTE2 | 6,014 | 5,372 | 4,884 | 3,002 | 2,540 |

The traced atomic row path improves by 253 cycles (1.82%). The dominant bottleneck remains MTE2, but MTE2 busy cycles drop 3.4%, VEC wait 4.2%, and PUSHQ 1.0% while preserving baseline address generation.

## Hardware verification (`remote_verify`)

Correctness: `UNIT_TEST PASS`; optimized persistent dispatch test `persistent_65536x1 PASS` with baseline providers pre-skipped by `grid_guard`.

| label | PyTorch / ACL ms | Baseline Triton1 ms | Baseline Triton2 ms | Optimized Triton ms | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small_atomic_8x256 | 0.023017 | 0.004970 | 0.003991 | 0.003655 | 1.360x |
| medium_atomic_128x4096 | 0.046245 | 0.018562 | 0.013386 | 0.016930 | 1.096x |
| irregular_atomic_17x3000 | 0.025858 | 0.007537 | 0.006690 | 0.005052 | 1.492x |
| exact_32768x8192 | 14.874719 | 3.970940 | 2.099750 | 3.964746 | 1.002x |

The atomic scalar-output path is used for the first three labels and removes the second `out.mean()` launch. The exact target shape uses the direct row path because hardware showed the atomic path regressed at `B=32768`; direct dispatch preserves the baseline per-row kernel and remains slightly faster than Baseline Triton1.
