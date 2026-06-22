# Performance Report

## Cannsim setup

- Baseline kernel: `_max_reduce_dim1_kernel` from `49_Max_reduction_over_a_dimension.py`
- Optimized kernel: `_max_reduce_dim1_kernel` from `opt_49_Max_reduction_over_a_dimension.py`
- Sub-kernel shape: `B=1, M=32, N=128`, `grid=(1,)`, fp32 input/output
- Target: `Ascend950`; conversion uses `0.4 ns/cycle`

## Cannsim trace comparison

| Kernel | wall_cycles | hardware time (ns) | x_events | i_events | bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline | 6,721 | 2,688.4 | 2,128 | 127 | MTE2 (3,709 busy cycles) |
| Optimized | 3,089 | 1,235.6 | 768 | 14 | MTE3 (2,509 busy cycles) |
| Delta | -54.0% | -54.0% | -63.9% | -89.0% | shifted from load/control to store tail |

## Pipeline details

| Kernel | MTE2 busy | MTE3 busy | VEC busy | SCALAR busy | SCALARLDST busy | PUSHQ busy | JUMPC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3,709 | 1,249 | 2,826 | 1,407 | 2,990 | 1,656 | 71 |
| Optimized | 1,016 | 2,509 | 1,008 | 578 | 479 | 1,375 | 2 |

## Interpretation

The optimized `[32,128]` tile reduction removes most scalar row-loop overhead from the baseline row-streaming path. `JUMPC` drops from 71 to 2 and wall cycles drop by 54.0%; the remaining bottleneck is mostly the unavoidable output store/wait tail for this one-tile sub-kernel.

## Hardware benchmark

`remote_verify` passed correctness for all providers and all dispatch-path tests (`dim=0`, `dim=1`, `dim=2`, negative dim alias, and benchmark shapes).

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 | Speedup vs Baseline2 |
|---|---:|---:|---:|---:|---:|---:|
| small_dim1 | 0.006834 | 0.026733 | 0.037537 | 0.014283 | 1.87x | 2.63x |
| medium_dim1 | 0.106707 | 0.280932 | 0.250504 | 0.165970 | 1.69x | 1.51x |
| target_original | 5.453156 | 24.488056 | 17.077112 | 13.200398 | 1.86x | 1.29x |

Target hardware latency: optimized Triton `13.200398 ms` for `(128, 4096, 4095), dim=1`; PyTorch / ACL remains faster at `5.453156 ms`.
