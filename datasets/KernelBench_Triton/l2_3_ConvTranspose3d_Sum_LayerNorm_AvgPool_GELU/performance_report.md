# Performance Report

## Cannsim setup

`cannsim_local_run` was run on a one-row LayerNorm sub-kernel probe (`M=64`, `ROWS_PER_CTA=1`, `grid=(1,1,1)`) for both the editable baseline and optimized Triton tiny path.  A 64-row probe was attempted first but was reduced because cannsim did not become safe within the timeout window.  Hardware latency uses `cycles * 0.4 ns`.

## Cannsim trace summary

| Path | Wall cycles | HW latency (us) | MTE2 | SCALAR | SCALARLDST | PUSHQ | VEC | MTE3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Triton LN row1 | 5251 | 2.100 | 3966 | 3777 | 3589 | 2176 | 821 | 857 |
| Optimized Triton LN row1 | 5279 | 2.112 | 3980 | 3685 | 3586 | 2202 | 825 | 869 |

The row-level LayerNorm trace is effectively unchanged, as expected: the main production optimization is host/dispatch selection plus removal of a dead host synchronization.  The optimized default path uses ACL/native post-processing and therefore has no custom Triton post-kernel trace for the default shape.

## Top cannsim instructions

| Path | Instruction | Cycles | Count |
|---|---|---:|---:|
| Baseline Triton LN row1 | `MOV_SRC_TO_DST_ALIGNv2` | 2730 | 4 |
| Baseline Triton LN row1 | `LD_XD_XN_IMM` | 2279 | 9 |
| Baseline Triton LN row1 | `VF` | 2152 | 6 |
| Baseline Triton LN row1 | `LDP_XI_XJ_XN` | 2036 | 4 |
| Baseline Triton LN row1 | `MOV_SPR_XN` | 1565 | 5 |
| Optimized Triton LN row1 | `MOV_SRC_TO_DST_ALIGNv2` | 2739 | 4 |
| Optimized Triton LN row1 | `LD_XD_XN_IMM` | 2276 | 9 |
| Optimized Triton LN row1 | `VF` | 2178 | 6 |
| Optimized Triton LN row1 | `LDP_XI_XJ_XN` | 1947 | 4 |
| Optimized Triton LN row1 | `MOV_SPR_XN` | 1570 | 5 |

## Remote hardware latency (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 | Speedup vs ACL |
|---|---:|---:|---:|---:|---:|---:|
| tiny_triton_path | 0.319749 | 0.569217 | 0.789990 | 0.488683 | 1.165x | 0.654x |
| medium_acl_path | 0.794584 | 3.746531 | 3.997688 | 0.764638 | 4.900x | 1.039x |
| default_required | 93.553238 | inf (pre-skipped grid cap) | inf (pre-skipped grid cap) | 79.572739 | n/a | 1.176x |

Correctness: `UNIT_TEST PASS`; optimized max absolute error was `1.38164e-4` on tiny Triton path, `1.84774e-6` on medium ACL path, and `3.01003e-6` on the required default shape.
