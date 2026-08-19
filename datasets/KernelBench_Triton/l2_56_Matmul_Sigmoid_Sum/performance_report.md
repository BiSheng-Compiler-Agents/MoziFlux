# Performance Report

## Verification artifacts

- Baseline cannsim trace: `/tmp/cannsim_local/l2_56_baseline/cannsim_20260629234228_test_kernel/report/trace_core0.json`
- Optimized cannsim trace: `/tmp/cannsim_local/l2_56_optimized/cannsim_20260629234605_test_kernel/report/trace_core0.json`
- Remote hardware verification: `remote_verify` passed correctness and benchmark.

## Cannsim sub-kernel trace comparison

The baseline sub-kernel computes one `(1 row × 64 hidden × 64 K)` tile with vector multiply/reduce. The optimized sub-kernel computes one `(16 rows × 64 hidden × 64 K)` Cube tile. Absolute cycles therefore are not directly comparable; normalized per input row is the relevant A/B metric.

| Kernel | Work tile | wall_cycles | hw_time_ns (`cycles*0.4`) | normalized ns / row | x_events | Dominant pipeline |
|---|---:|---:|---:|---:|---:|---|
| Baseline vector GEMM | 1×64×64 | 4,696 | 1,878.4 | 1,878.4 | 715 | SCALAR 1,981 cyc |
| Optimized Cube GEMM | 16×64×64 | 5,973 | 2,389.2 | 149.3 | 1,985 | SCALARLDST 2,569 cyc |

Normalized sub-kernel speedup: `1878.4 / 149.3 = 12.58x` per row for the GEMM tile.

### Pipeline table

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Notes |
|---|---:|---:|---|
| SCALAR | 1,981 | 2,385 | Baseline scalar-bound; optimized scalar cost amortized over 16 rows. |
| SCALARLDST | 1,751 | 2,569 | Optimized stores a 16×64 logits tile. |
| MTE2 | 1,250 | 1,313 | Similar input traffic for much more math. |
| VEC | 788 | 1,286 | Mostly waits/data motion around stores. |
| RVECEX | 333 | 12 | Vector arithmetic for GEMM is eliminated. |
| CUBE | 0 | 434 | Cube is activated by `tl.dot`. |
| MTE3 | 518 | 2,302 | Optimized writes full logits tile; second kernel reduces sigmoid. |

### Top instruction bottlenecks

| Kernel | Critical instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---|---:|---:|---:|
| Baseline | `LDP_XI_XJ_XN` | SCALAR | 6 | 2,963 | 494 |
| Baseline | `MOV_SRC_TO_DST_ALIGNv2` | MTE2 | 3 | 2,217 | 739 |
| Baseline | `RV_VCADD` | RVECEX | 65 | 1,430 | 22 |
| Optimized | `ST_XD_XN_IMM` | SCALARLDST | 22 | 26,051 | 1,184 |
| Optimized | `RV_VSTI` | RVECST | 640 | 7,708 | 12 |
| Optimized | `MOV_SPR_XN` | MTE2 | 9 | 6,841 | 760 |

## Remote hardware latency

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| tiny_irregular | 0.088836 | 0.148744 | 0.161041 | 0.091784 | 1.62x |
| small_aligned | 0.151457 | 1.783258 | 1.791307 | 0.236409 | 7.54x |
| medium_largeK | 0.763062 | 24.852236 | 24.251120 | 2.961900 | 8.39x |
| default | 47.104095 | 4728.518555 | 4454.611816 | 702.049500 | 6.74x |

Correctness: optimized path passed all unit shapes; default max_abs was `0.00292969` against PyTorch/ACL. Baseline comparison providers exceeded the stricter default comparison tolerance but are read-only/reference comparisons for this task.
