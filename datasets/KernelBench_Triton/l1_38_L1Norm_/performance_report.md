# Performance Report

## Summary

- Correctness: `remote_verify` passed all providers and all benchmark shapes, including `target=(32768, 65535)`.
- Target hardware latency: optimized Triton `14.871285 ms`; baseline Triton1 `14.926386 ms`; PyTorch / ACL `29.167211 ms`.
- Target speedup: `1.004x` vs Baseline Triton1 and `1.961x` vs PyTorch / ACL. Geomean speedup vs Baseline Triton1 across all benchmark rows: `1.019x`.

## Hardware benchmark (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs B1 |
|---|---:|---:|---:|---:|---:|
| tiny_block64 | 0.028578 | 0.034433 | 0.032001 | 0.034874 | 0.987x |
| small_block1024 | 0.036347 | 0.033910 | 0.034465 | 0.033944 | 0.999x |
| mid_block2048 | 0.034256 | 0.036295 | 0.034048 | 0.033934 | 1.070x |
| large_block4096 | 0.037751 | 0.035410 | 0.034836 | 0.034109 | 1.038x |
| target | 29.167211 | 14.926386 | 17.031329 | 14.871285 | 1.004x |

## Cannsim setup

Sub-kernel: `B=1, N=16384, BLOCK_SIZE=4096`, fp32 input initialized to 1.0, expected output `1/N`. Both hosts passed correctness. The baseline cannsim compile uses the baseline arithmetic/tiling with `.cg` omitted for simulator compatibility; hardware benchmark above uses the unmodified `38_L1Norm_.py` provider.

| Variant | trace_core0.json | wall_cycles | hardware time (cycles × 0.4 ns) | x_events | i_events |
|---|---|---:|---:|---:|---:|
| Baseline | `/tmp/cannsim_local/l1_38_l1norm_baseline/cannsim_20260624213914_test_kernel/report/trace_core0.json` | 9768 | 3.907 us | 2559 | 101 |
| Optimized | `/tmp/cannsim_local/l1_38_l1norm_optimized/cannsim_20260624214126_test_kernel/report/trace_core0.json` | 9757 | 3.903 us | 2614 | 90 |

## Cannsim pipeline utilization

### Baseline

| pipeline | ops | busy_cyc | lane_sum | lanes | window | note |
|---|---:|---:|---:|---:|---|---|
| 04_MTE2 | 36 | 4197 | 8833 | 11 | [5689,13668] | BOTTLENECK |
| 05_VEC | 12 | 4102 | 7578 | 8 | [5845,12155] | |
| 10_PUSHQ | 24 | 2495 | 2529 | 2 | [5845,12278] | |
| 07_MTE3 | 8 | 2412 | 5409 | 4 | [11245,13658] | |
| 02_SCALARLDST | 90 | 2265 | 31061 | 24 | [3927,11427] | |
| 12_RVECEX | 1044 | 1452 | 10876 | 17 | [6867,12259] | |
| 01_SCALAR | 547 | 1262 | 5018 | 10 | [3907,13671] | |
| 13_RVECLD | 524 | 1076 | 4720 | 9 | [6866,12254] | |
| 14_RVECST | 264 | 360 | 2376 | 9 | [7135,12266] | |

### Optimized

| pipeline | ops | busy_cyc | lane_sum | lanes | window | note |
|---|---:|---:|---:|---:|---|---|
| 04_MTE2 | 36 | 4843 | 8000 | 8 | [5749,12653] | BOTTLENECK |
| 05_VEC | 12 | 4480 | 6592 | 4 | [5856,12454] | |
| 10_PUSHQ | 26 | 2812 | 2964 | 3 | [5720,12652] | |
| 07_MTE3 | 8 | 2482 | 4479 | 4 | [11162,13645] | |
| 02_SCALARLDST | 52 | 2096 | 6641 | 12 | [3920,12129] | |
| 12_RVECEX | 1048 | 1589 | 10911 | 17 | [5843,12595] | |
| 13_RVECLD | 781 | 1209 | 7493 | 14 | [5842,12590] | |
| 01_SCALAR | 376 | 1006 | 2724 | 10 | [3901,13654] | |
| 14_RVECST | 265 | 665 | 4677 | 9 | [5850,12640] | |

## Top cannsim instruction changes

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| wall_cycles | 9768 | 9757 | -11 |
| i_events | 101 | 90 | -11 |
| SCALARLDST ops | 90 | 52 | -38 |
| SCALAR ops | 547 | 376 | -171 |
| `ST_XD_XN_IMM` total_cyc | 27886 | 4901 | -22985 |
| `LD_XD_XN_IMM` total_cyc | 3016 | not top-12 | reduced below top list |

The trace confirms the intended scalar-spill reduction; total sub-kernel wall time is nearly unchanged because the row-normalization kernel remains GM/MTE bound.
