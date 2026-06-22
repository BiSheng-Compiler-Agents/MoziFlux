# Performance Report

## Cannsim setup

- Baseline sub-kernel: `_softplus_kernel`, `BLOCK_SIZE=4096`, `N=4096`, grid `(1,)`.
- Optimized sub-kernel: `_softplus_direct_kernel`, `BLOCK_SIZE=8192`, `N=8192`, grid `(1,)`.
- cannsim jobs: `l1_29_softplus_baseline`, `l1_29_softplus_optimized`.
- Conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim trace comparison

| Kernel | Elements/tile | Wall cycles | Normalized cycles / 4096 elems | Est. time / tile (ns) | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline Triton1 | 4096 | 3877 | 3877.0 | 1550.8 | MTE3 WAIT_FLAG_VEC |
| Optimized Triton | 8192 | 4681 | 2340.5 | 1872.4 | MTE3 WAIT_FLAG_VEC / RVECEX |

## Pipeline table

| Kernel | MTE3 busy | SCALAR busy | MTE2 busy | VEC busy | RVECEX busy | RVECLD busy | RVECST busy | PUSHQ busy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 2098 | 1777 | 1011 | 1005 | 624 | 534 | 450 | 670 |
| Optimized | 2900 | 1779 | 1068 | 1062 | 1203 | 1086 | 925 | 1249 |

## Top trace instructions

| Kernel | Top instruction | Cycles | Notes |
|---|---|---:|---|
| Baseline | WAIT_FLAG_VEC @ MTE3 | 1656 | Store-side wait dominates the 4096-element tile. |
| Baseline | LD_XD_XN_IMM @ SCALARLDST | 1218 | Scalar setup cost is large relative to tile work. |
| Optimized | RV_VLN @ RVECEX | 2304 | More work per tile from 8192 elements. |
| Optimized | WAIT_FLAG_VEC @ MTE3 | 2292 | Store-side wait remains bottleneck, but per-element cost is lower. |

## Hardware benchmark (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_irregular | 0.032888 | 0.057034 | 0.050260 | 0.049030 |
| direct_medium | 0.058365 | 0.052394 | 0.052438 | 0.050148 |
| persistent_original | 47.358231 | inf (grid overflow skipped) | 0.055044 | 0.030437 |

Correctness: `UNIT_TEST PASS`; optimized max absolute error was `2.38419e-07` on direct-path tests and `0` on the oversized original-shape test reported by the verifier.
