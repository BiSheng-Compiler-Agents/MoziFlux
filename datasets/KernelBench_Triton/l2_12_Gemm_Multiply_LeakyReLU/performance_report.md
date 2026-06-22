# Performance Report

## Cannsim setup

- Baseline sub-kernel: `_linear_mul_leaky_kernel`, `M=N=128`, `K=64`, `BLOCK_M=128`, `BLOCK_N=128`, `BLOCK_K=64`, grid `(1, 1)`.
- Optimized Triton sub-kernel: `_linear_mul_leaky_kernel_auto` body, same tile and K extent, grid `(1, 1)`.
- Both cannsim records completed the user application and generated `trace_core0.json`. `cannsim_local_run` flagged instr quiet-time after record completion, so `cannsim report -n 0` was run manually against the completed record directories.

## Cannsim trace summary

| Metric | Baseline | Optimized Triton path | Delta |
|---|---:|---:|---:|
| wall_cycles | 15,425 | 15,437 | +0.1% |
| hardware time @0.4 ns/cycle | 6.170 µs | 6.175 µs | +0.005 µs |
| x_events | 7,667 | 7,646 | -0.3% |
| i_events | 226 | 226 | 0.0% |
| npubin size | 17,888 B | 17,992 B | +0.6% |

## Cannsim pipeline utilization

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Delta |
|---|---:|---:|---:|
| MTE3 | 7,137 | 7,010 | -1.8% |
| MTE2 | 5,267 | 5,230 | -0.7% |
| FLOWCTRL | 5,146 | 5,083 | -1.2% |
| FIXP | 5,005 | 5,005 | 0.0% |
| PUSHQ | 4,760 | 4,698 | -1.3% |
| MTE1 | 4,453 | 4,453 | 0.0% |
| CUBE | 4,448 | 4,448 | 0.0% |
| RVECST | 4,037 | 4,037 | 0.0% |
| RVECLD | 3,742 | 3,742 | 0.0% |
| SCALARLDST | 2,761 | 2,794 | +1.2% |
| VEC | 2,617 | 2,594 | -0.9% |
| SCALAR | 2,509 | 1,476 | -41.2% |
| RVECSU | 1,123 | 1,123 | 0.0% |
| RVECEX | 720 | 720 | 0.0% |

## Top instruction costs

| Instruction | Pipeline | Baseline total_cyc | Optimized total_cyc |
|---|---|---:|---:|
| RV_VSTI | RVECST | 30,613 | 30,613 |
| ST_XD_XN_IMM | SCALARLDST | 21,550 | 28,918 |
| RV_VLDI | RVECLD | 20,754 | 20,754 |
| MOV_SPR_XN | MTE2 | 9,294 | 8,845 |
| WAIT_FLAG_CUBE | MTE1 | 8,865 | 8,865 |
| WAIT_FLAG_VEC | MTE3 | 8,577 | 8,394 |
| SET_INTRA_BLOCKI | FLOWCTRL | 5,421 | 5,358 |
| VF | PUSHQ | 4,778 | 4,716 |
| MMAD | CUBE | 4,163 | 4,163 |

## Hardware latency (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small | 0.039928 | 0.064827 | 0.066225 | 0.038368 | 1.690x |
| irregular | 0.102847 | 0.170135 | 0.171918 | 0.102637 | 1.658x |
| target | 6.425581 | 6.035031 | 6.022677 | 6.043127 | 0.999x |
| geomean | — | — | — | — | 1.409x |

## Interpretation

The custom Triton sub-kernel remains dominated by MTE3/FIXP/Cube waits, so micro-optimizing the target tile gives little single-tile gain in cannsim. The measured win comes from host dispatch: ACL is faster for small/medium GEMM shapes, while the large target keeps the existing autotuned Triton path because it is faster than PyTorch / ACL.
