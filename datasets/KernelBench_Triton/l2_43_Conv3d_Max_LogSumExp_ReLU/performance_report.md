# Performance Report

## Cannsim setup

- Baseline trace: `_lse_relu_reduce_c_kernel`, scale-limited to `C=2, BLOCK=4` because the production-like `C=64, BLOCK=64` baseline failed BiSheng compilation with VF stack spill (`23328 > 6144` bytes). A prior `C=8, BLOCK=16` trace also ran too long for reliable report generation.
- Optimized trace: `_lse_relu_lastdim_kernel`, production channel tile `C=64, BLOCK_M=16, BLOCK_C=64`.
- Both runs used `cannsim_local_run(..., gen_report=True)` and `trace_core0.json`; hardware time uses `cycles * 0.4 ns`.

## Cannsim trace comparison

| Kernel | Micro-probe | wall_cycles | HW time (ns) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline strided LSE | C=2, BLOCK=4 | 6,645 | 2,658 | 1,024 | 103 | PUSHQ 3,508 cycles |
| Optimized contiguous LSE | C=64, BLOCK_M=16 | 3,179 | 1,272 | 450 | 14 | MTE3 1,389 cycles |

## Pipeline utilization

| Kernel | Pipeline | Ops | Busy cycles | Notes |
|---|---|---:|---:|---|
| Baseline | PUSHQ | 35 | 3,508 | VF dispatch dominates even at tiny C=2/BLOCK=4 |
| Baseline | SCALARLDST | 73 | 2,516 | many scalar loads/stores from loop-carried strided reduction |
| Baseline | SCALAR | 695 | 1,446 | decode/div/mod and pointer arithmetic overhead |
| Baseline | RVECEX | 163 | 338 | useful vector math is a small share of total time |
| Optimized | MTE3 | 2 | 1,389 | output/writeback wait is dominant after vectorization |
| Optimized | SCALARLDST | 2 | 1,228 | fixed kernel argument/setup cost |
| Optimized | MTE2 | 3 | 965 | contiguous tile load |
| Optimized | VEC | 1 | 959 | exp/log reduction support |
| Optimized | PUSHQ | 3 | 314 | much lower dispatch pressure |

## Critical instructions

| Kernel | Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---|---:|---:|---:|
| Baseline | VF | PUSHQ | 17 | 3,593 | 211 |
| Baseline | ST_XD_XN_IMM | SCALARLDST | 34 | 3,218 | 95 |
| Baseline | LDP_XI_XJ_XN | SCALAR | 6 | 1,492 | 249 |
| Optimized | LDP_XI_XJ_XN | SCALAR | 3 | 1,432 | 477 |
| Optimized | WAIT_FLAG_VEC | MTE3 | 1 | 1,254 | 1,254 |
| Optimized | ST_XD_XN_IMM | SCALARLDST | 1 | 1,226 | 1,226 |
| Optimized | LD_XD_XN_IMM | SCALARLDST | 1 | 1,221 | 1,221 |

## Remote hardware latency

`remote_verify` correctness passed; comparison baselines were kept visible but pre-skipped because they fail with `MLIRCompilationError` on all tested shapes.

| label | PyTorch / ACL (ms) | Baseline Triton1 | Baseline Triton2 | Optimized Triton (ms) | ACL / Optimized |
|---|---:|---:|---:|---:|---:|
| small | 0.642187 | inf | inf | 0.453910 | 1.415x |
| medium | 4.393136 | inf | inf | 4.665996 | 0.942x |
| default | 57.263958 | inf | inf | 60.777836 | 0.942x |

Conclusion: the optimized Triton reduction is correct and improves the custom strided reduction trace shape, but full medium/default hardware latency is slower than pure ACL because the `permute(...).contiguous()` materialization offsets the faster contiguous Triton reduction.
