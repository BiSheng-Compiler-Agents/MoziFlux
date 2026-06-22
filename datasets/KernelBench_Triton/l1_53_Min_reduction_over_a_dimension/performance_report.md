# Performance Report

## Cannsim setup

- Baseline sub-kernel: one `(b,n)` output, `M=128,N=1,BLOCK_K=64`, trace `/tmp/cannsim_local/l1_53_min_baseline_bm128/cannsim_20260625031635_test_kernel/report/trace_core0.json`.
- Optimized sub-kernel: one N tile, `M=128,N=128,BLOCK_M=128,BLOCK_N=128`, trace `/tmp/cannsim_local/l1_53_min_optimized_bm128/cannsim_20260625031817_test_kernel/report/trace_core0.json`.
- Hardware conversion: `cycles * 0.4 ns`.

## Cannsim trace summary

| Kernel | Logical work | Wall cycles | HW time | Bottleneck | Busy cycles | Top critical instruction |
|---|---:|---:|---:|---|---:|---|
| Baseline dim=1 | 1 output | 3,659 | 1,463.6 ns | SCALARLDST | 1,955 | `ST_XD_XN_IMM` 5,889 total cycles |
| Baseline normalized | 128 outputs | 468,352 | 187,340.8 ns | SCALARLDST | 250,240 | scalar-column repetition |
| Optimized dim=1 tile | 128 outputs | 8,700 | 3,480.0 ns | MTE3 | 5,493 | `WAIT_FLAG_VEC` 5,150 cycles |

## Pipeline table

| Kernel | SCALARLDST | MTE2 | VEC | SCALAR | PUSHQ | MTE3 | RVECEX | RVECLD/RVECST |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (1 output) | 1,955 | 1,115 | 1,108 | 706 | 648 | 528 | 93 | 64 |
| Baseline normalized (128 outputs) | 250,240 | 142,720 | 141,824 | 90,368 | 82,944 | 67,584 | 11,904 | 8,192 |
| Optimized (128 outputs) | 2,456 | 1,329 | 577 | 913 | 5,240 | 5,493 | 2,600 | 5,798 |

## Interpretation

The optimized tile computes 128 output columns in 8,700 cycles versus 3,659 cycles for one scalar baseline column; normalized to the same 128-output work unit, cannsim indicates `468,352 -> 8,700` cycles (`53.8x`). The main win is replacing per-column strided reduction with a contiguous N-tile reduction and legal target grid.

## Hardware latency (`remote_verify`)

Correctness passed for optimized dim=0, dim=1 target, and dim=2 paths.

| label | PyTorch / ACL (ms) | Baseline Triton1 | Baseline Triton2 | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| small_dim1 | 0.006880 | inf | inf | 0.014162 |
| target_dim1 | 5.445765 | inf | inf | 9.811450 |
| dim0_path | 0.004157 | inf | inf | 0.009858 |
| dim2_path | 0.005477 | inf | inf | 0.040802 |

Baseline Triton providers are preserved in the profiler but pre-skipped (`provider_guard`) because the input provider uses unsafe Ascend dispatch/cache patterns at target scale and `base_*.py` is read-only.
