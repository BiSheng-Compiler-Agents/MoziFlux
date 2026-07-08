# Performance Report

## Verification summary

- `cannsim_local_run` baseline-compatible epilogue trace: `/tmp/cannsim_local/l2_53_baseline/cannsim_20260629225755_test_kernel/report/trace_core0.json`
- `cannsim_local_run` optimized epilogue trace: `/tmp/cannsim_local/l2_53_optimized/cannsim_20260629225942_test_kernel/report/trace_core0.json`
- Remote hardware verification: `UNIT_TEST PASS`; benchmark completed.

The cannsim baseline sub-kernel intentionally removes the source file's non-semantic `cache_modifier=".cg"` so the baseline tile can compile under Triton-Ascend; the tile size and exact scale + Hardtanh + GELU math match the input baseline epilogue.

## Cannsim trace comparison

| Kernel | Elements in sub-kernel | Wall cycles | Cycles / element | Elements / cycle | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline-compatible epilogue (`BLOCK_N=1024`) | 1,024 | 3,867 | 3.776 | 0.265 | MTE3 store wait |
| Optimized epilogue (`BLOCK_SIZE=4096`) | 4,096 | 5,898 | 1.440 | 0.694 | MTE3 store wait / vector issue |

Normalized cannsim throughput improved **2.62x** (`0.694 / 0.265` elements/cycle). Hardware time uses `cycles * 0.4 ns`: baseline tile `1.547 us`, optimized tile `2.359 us`; normalized hardware time is `1.511 ns/element` vs `0.576 ns/element`.

## Cannsim pipeline table

| Pipeline | Baseline busy cycles | Optimized busy cycles | Notes |
|---|---:|---:|---|
| MTE3 | 2,012 | 4,071 | Dominant store path; higher absolute cycles because optimized tile has 4x elements. |
| PUSHQ | 767 | 2,687 | Larger tile issues a larger vector packet. |
| RVECEX | 711 | 2,631 | Scales with element count; normalized vector work is lower. |
| SCALAR | 1,864 | 1,833 | Essentially flat despite 4x elements, showing reduced scalar overhead/element. |
| SCALARLDST | 1,739 | 1,726 | Essentially flat despite 4x elements. |
| MTE2 | 964 | 1,015 | Nearly flat due contiguous larger DMA. |
| VEC | 959 | 1,010 | Nearly flat. |
| RVECLD | 144 | 576 | Scales with 4x elements. |
| RVECST | 144 | 576 | Scales with 4x elements. |

## Hardware benchmark latency (ms)

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| small | 0.346292 | 0.362505 | 0.332907 | 0.317968 | 1.140x |
| irregular | 0.401919 | 0.526113 | 0.389573 | 0.360715 | 1.459x |
| target | 32.898743 | 36.758179 | 33.551254 | 33.437344 | 1.099x |

Geomean speedup vs Baseline Triton1: **1.222x**. Target optimized latency is faster than editable Baseline Triton1 and the read-only Baseline Triton2, but slower than pure PyTorch/ACL because the GEMM dominates total latency.
