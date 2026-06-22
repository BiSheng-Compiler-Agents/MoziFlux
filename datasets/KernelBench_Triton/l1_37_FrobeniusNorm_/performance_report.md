# Performance Report

## Summary

- Hardware verification: `UNIT_TEST PASS` on all optimized dispatch paths.
- Target hardware latency: PyTorch / ACL `25.516468 ms`, Optimized Triton `16.953049 ms` (`1.51x` faster than PyTorch / ACL).
- Baseline Triton1 and Baseline Triton2 are skipped at the target shape because their direct grid would launch `114,688` programs, exceeding Ascend `coreDim <= 65535`.

## Hardware benchmark (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| small_1k | 0.007084 | 0.005059 | 0.005737 | 0.004803 |
| medium_1m | 0.019126 | 0.013868 | 0.014943 | 0.014331 |
| target_112x64x512x512 | 25.516468 | inf | inf | 16.953049 |

## Cannsim setup

Sub-kernel simulation used `N=8192`, `BLOCK=8192`, `grid=(1,)`, fp32 input values of `1.0`, and correctness checked `y = 1 / sqrt(8192)`. `cannsim report` generated `trace_core0.json`; record logs show the known post-trace simulator teardown abort, but trace reports were produced successfully.

## Cannsim trace comparison

| metric | Baseline two-launch sub-kernel | Optimized overflow-path sub-kernel |
|---|---:|---:|
| trace path | `/tmp/cannsim_local/cannsim_baseline_1782336367/cannsim_20260624212630_test_kernel/report/trace_core0.json` | `/tmp/cannsim_local/cannsim_optimized_1782336477/cannsim_20260624212824_test_kernel/report/trace_core0.json` |
| wall_cycles | 3,432 | 9,469 |
| hardware time (`cycles * 0.4ns`) | 1.373 µs | 3.788 µs |
| x_events | 636 | 896 |
| i_events | 12 | 44 |
| dominant pipeline | SCALAR, 1,822 busy cycles | SCALAR, 2,413 busy cycles |
| next bottlenecks | SCALARLDST 1,743; MTE2 1,075; VEC 1,044 | SCALARLDST 2,220; PUSHQ 1,714; MTE2 1,434 |

## Baseline pipeline table

| pipeline | ops | busy_cyc | lane_sum | lanes |
|---|---:|---:|---:|---:|
| SCALAR | 105 | 1,822 | 2,628 | 9 |
| SCALARLDST | 7 | 1,743 | 3,181 | 4 |
| MTE2 | 3 | 1,075 | 1,075 | 1 |
| VEC | 1 | 1,044 | 1,044 | 1 |
| PUSHQ | 2 | 550 | 550 | 1 |
| RVECEX | 386 | 504 | 4,748 | 19 |
| RVECLD | 129 | 425 | 1,161 | 9 |
| RVECST | 1 | 9 | 9 | 1 |
| RVECSU | 1 | 1 | 1 | 1 |
| MTE3 | 1 | 1 | 1 | 1 |

## Optimized overflow-path pipeline table

| pipeline | ops | busy_cyc | lane_sum | lanes |
|---|---:|---:|---:|---:|
| SCALAR | 261 | 2,413 | 5,349 | 10 |
| SCALARLDST | 15 | 2,220 | 2,990 | 3 |
| PUSHQ | 8 | 1,714 | 1,718 | 2 |
| MTE2 | 11 | 1,434 | 1,618 | 2 |
| VEC | 2 | 1,245 | 1,245 | 1 |
| MTE3 | 4 | 683 | 683 | 1 |
| RVECEX | 424 | 611 | 5,249 | 19 |
| RVECLD | 148 | 461 | 1,333 | 9 |
| RVECST | 19 | 51 | 171 | 9 |
| FLOWCTRL | 4 | 14 | 18 | 2 |

## Interpretation

At sub-kernel scale, the overflow path has extra launch/reduction overhead, so it is slower than the two-launch direct path. At the real target shape, the direct baseline is invalid due grid overflow; the optimized persistent partial path is the only Triton path that passes and reaches `16.953049 ms`.
