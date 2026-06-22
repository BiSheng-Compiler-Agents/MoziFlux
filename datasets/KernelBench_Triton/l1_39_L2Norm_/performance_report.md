# Performance Report

## Correctness / hardware verification

Remote Ascend verification passed (`UNIT_TEST PASS`). Max absolute optimized error was <= `1.192093e-07` over small, irregular, row-overflow, and target shapes.

## Hardware latency (`profile_kernels.py`, ms)

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton |
|---|---:|---:|---:|---:|
| small_singlepass | 0.010601 | 0.011600 | 0.011024 | 0.004830 |
| irregular_medium | 0.050221 | 0.066155 | 0.056074 | 0.055870 |
| overflow_rows | 0.016581 | inf | inf | 1.201293 |
| target_32768x65535 | 17.361588 | 15.572003 | 17.040531 | 15.493136 |

Target speedup vs Baseline Triton1: `15.572003 / 15.493136 = 1.0051x`. Small-row speedup vs Baseline Triton1: `2.40x`.

## cannsim trace comparison

Baseline trace: `/tmp/cannsim_local/l1_39_l2norm_baseline/cannsim_20260624215118_test_kernel/report/trace_core0.json`
Optimized trace: `/tmp/cannsim_local/l1_39_l2norm_opt_final3/cannsim_20260624220431_test_kernel/report/trace_core0.json`

| kernel | wall cycles | x events | i events | bottleneck |
|---|---:|---:|---:|---|
| baseline rowwise | 4553 | 970 | 39 | SCALARLDST 1975 busy cycles |
| optimized final large-row | 5329 | 885 | 43 | SCALARLDST 1868 busy cycles |

| kernel | SCALAR | SCALARLDST | MTE2 | MTE3 | VEC | PUSHQ | RVECEX | RVECLD | RVECST |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 765 | 1975 | 1364 | 1280 | 1248 | 1595 | 600 | 472 | 295 |
| optimized final large-row | 731 | 1868 | 1376 | 1136 | 1353 | 1471 | 606 | 481 | 304 |

The final grid-cap-safe optimized trace has higher sub-kernel wall cycles because the persistent row loop adds control overhead that is visible at `grid=1`. Hardware target latency is still slightly faster from simpler contiguous indexing, while the small-row path has the large measured win.

## Top cannsim instructions

| kernel | top instruction | total cycles | note |
|---|---|---:|---|
| baseline | ST_XD_XN_IMM | 2196 | scalar local store pressure |
| baseline | RV_VLDI | 1799 | vector load setup |
| optimized | ST_XD_XN_IMM | 2557 | row-loop scalar store pressure |
| optimized | RV_VLDI | 1808 | vector load setup |
