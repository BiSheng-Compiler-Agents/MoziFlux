# Performance Report

## Correctness and hardware latency

Remote Ascend verification passed (`UNIT_TEST PASS`). `base_*.py` was not read because the sandbox marks reference files read-only/no-read; the Baseline Triton2 column is kept parser-visible with `inf`/`SKIP_UNAVAILABLE`.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 | Optimized Triton (ms) | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| tiny_direct | 0.288679 | 0.382169 | inf | 0.367147 | 1.04x |
| irregular_direct | 0.299384 | 0.396965 | inf | 0.381412 | 1.04x |
| default_required | 90.400230 | 85.358635 | inf | 1.871922 | 45.60x |

Default-shape optimized latency is **1.871922 ms** on hardware, **45.60x** faster than Baseline Triton1 and **48.29x** faster than PyTorch / ACL.

## Cannsim trace comparison

Sub-kernel probes used `grid=(1,)`, `L=2048`, `BLOCK=2048`, fp32 input, and one reduced row. These traces compare the old output-plane GlobalAvgPool reduction against the new input-plane spatial-sum Triton reduction. Full-shape algorithmic savings come from reducing 16,777,216 input elements instead of materializing/reducing 53,084,160 ConvTranspose output elements.

| Kernel | trace_core0.json | wall cycles | hardware ns (`cycles*0.4`) | x events | i events | bottleneck |
|---|---|---:|---:|---:|---:|---|
| Baseline GAP | `/tmp/cannsim_local/l2_77_baseline_gap/cannsim_20260630061515_test_kernel/report/trace_core0.json` | 3,673 | 1,469.2 | 216 | 16 | SCALAR |
| Optimized spatial sum | `/tmp/cannsim_local/l2_77_optimized_spatial_sum/cannsim_20260630061701_test_kernel/report/trace_core0.json` | 3,078 | 1,231.2 | 194 | 12 | SCALAR |

## Pipeline tables

### Baseline GAP micro-kernel

| Pipeline | Ops | Busy cycles |
|---|---:|---:|
| SCALAR | 94 | 1813 |
| SCALARLDST | 5 | 1281 |
| MTE2 | 3 | 996 |
| VEC | 1 | 971 |
| PUSHQ | 4 | 743 |
| RVECEX | 68 | 163 |
| MTE3 | 2 | 102 |
| RVECLD | 34 | 60 |
| RVECST | 2 | 18 |
| FLOWCTRL | 2 | 7 |

Top critical instructions: `LDP_XI_XJ_XN` 1429 cycles, `LD_XD_XN_IMM` 1229, `STI_XN_IMM` 1228, `MOV_SRC_TO_DST_ALIGNv2` MTE2 989, `WAIT_FLAG_MTE2` 971.

### Optimized spatial-sum micro-kernel

| Pipeline | Ops | Busy cycles |
|---|---:|---:|
| SCALAR | 80 | 1799 |
| SCALARLDST | 3 | 1255 |
| MTE2 | 3 | 990 |
| VEC | 1 | 965 |
| PUSHQ | 2 | 196 |
| RVECEX | 66 | 150 |
| MTE3 | 2 | 101 |
| RVECLD | 33 | 51 |
| RVECST | 1 | 9 |
| FLOWCTRL | 2 | 7 |

Top critical instructions: `LDP_XI_XJ_XN` 1438 cycles, `LD_XD_XN_IMM` 1225, `STI_XN_IMM` 1224, `MOV_SRC_TO_DST_ALIGNv2` MTE2 983, `WAIT_FLAG_MTE2` 965.

## Interpretation

The micro-kernel trace improves wall cycles by **1.19x** and cuts PUSHQ events from 4 to 2. The large hardware win comes from the algorithmic shortcut: ConvTranspose output materialization and full-output BatchNorm/GlobalAvgPool are replaced by input spatial sums plus a tiny channel GEMM.
