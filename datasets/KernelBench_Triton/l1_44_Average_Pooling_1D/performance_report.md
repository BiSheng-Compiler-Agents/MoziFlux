# Performance Report

## Cannsim setup
- Baseline trace: `/tmp/cannsim_local/l1_44_avgpool_baseline_b32/cannsim_20260625003402_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l1_44_avgpool_optimized_cols_b32/cannsim_20260625004910_test_kernel/report/trace_core0.json`
- Sub-kernel: one row/output tile, `BLOCK=32`, `KERNEL_SIZE=8`; `BLOCK=256` timed out in cannsim stability detection, so the shrunken trace is used for bottleneck comparison.
- Hardware time conversion: `cycles * 0.4 ns`.

## Cannsim trace summary

### Baseline-equivalent

```text
=== /tmp/cannsim_local/l1_44_avgpool_baseline_b32/cannsim_20260625003402_test_kernel/report/trace_core0.json Profile ===
wall_cycles: 6908  |  x_events: 2791  |  i_events: 480
time_window: [3894,10802]

--- Pipeline Utilization ---
pipeline        ops  busy_cyc  lane_sum  lanes         window
02_SCALARLDST   513      5608     10781      6    [3915,9557]  ← BOTTLENECK
01_SCALAR      2194      3369     10690      9   [3894,10798]
07_MTE3           2      1208      1208      1   [9584,10793]
10_PUSHQ          4       717       724      3   [9578,10295]
12_RVECEX        66        65       433     11  [10211,10276]
13_RVECLD         8        14        77      8  [10210,10224]
14_RVECST         1         9         9      1  [10274,10283]
15_FLOWCTRL       2         7         9      2  [10795,10802]
04_MTE2           1         5         5      1    [9587,9592]

--- Top Instructions by Cycle Cost (top 12 of 47) ---
instruction             pipe        cnt  total_cyc  avg_cyc
ST_XD_XN_IMM            SCALARLDST  256       8480       33  ← CRITICAL
ADD                     SCALAR      772       3088        4
LD_XD_XN                SCALARLDST  256       1823        7
LDP_XI_XJ_XN            SCALAR        3       1433      478
ADD_IMM                 SCALAR      268       1072        4
SIGNEXT                 SCALAR      261       1044        4
MIN                     SCALAR      256       1024        4
MAX                     SCALAR      256       1024        4
CMP_IMM                 SCALAR      256       1024        4
VF                      PUSHQ         1        712      712
WAIT_FLAG_VEC           MTE3          1        712      712
MOV_SRC_TO_DST_ALIGNv2  MTE3          1        496      496

--- Worst Single Events (top 6 distinct, deduped) ---
VF@PUSHQ  ts=9583  dur=712
WAIT_FLAG_VEC@MTE3  ts=9584  dur=712
LD_XD_XN@SCALARLDST  ts=4427  dur=647  (×255 similar, avg ~5)
MOV_SRC_TO_DST_ALIGNv2@MTE3  ts=10297  dur=496
DC_PRELOAD_XN_IMM@SCALAR  ts=3898  dur=492
LDP_XI_XJ_XN@SCALAR  ts=3913  dur=478  (×2 similar, avg ~478)

--- Control Flow / Instant Events ---
JUMPC×256  JUMP×217  MOV_SPR_XN×2  NOP×2  WAIT_FLAG_SCALAR×1  BAR×1  END×1

```

### Optimized column-tile kernel

```text
=== /tmp/cannsim_local/l1_44_avgpool_optimized_cols_b32/cannsim_20260625004910_test_kernel/report/trace_core0.json Profile ===
wall_cycles: 6886  |  x_events: 2583  |  i_events: 482
time_window: [3883,10769]

--- Pipeline Utilization ---
pipeline        ops  busy_cyc  lane_sum  lanes         window
02_SCALARLDST   514      5345     11710      7    [3907,9301]  ← BOTTLENECK
01_SCALAR      1982      3148     10307      9   [3883,10765]
07_MTE3           2       880       880      1   [9879,10760]
10_PUSHQ          4       387       391      2   [9311,10257]
12_RVECEX        69        76       452     11  [10054,10238]
13_RVECLD         8        14        77      8  [10053,10067]
14_RVECST         1         9         9      1  [10236,10245]
15_FLOWCTRL       2         7         9      2  [10762,10769]
04_MTE2           1         5         5      1    [9896,9901]

--- Top Instructions by Cycle Cost (top 12 of 50) ---
instruction             pipe        cnt  total_cyc  avg_cyc
ST_XD_XN_IMM            SCALARLDST  256       8892       35  ← CRITICAL
ADD                     SCALAR      518       2072        4
LDP_XI_XJ_XN            SCALAR        4       1900      475
LD_XD_XN                SCALARLDST  256       1870        7
SIGNEXT                 SCALAR      275       1100        4
ADD_IMM                 SCALAR      270       1080        4
MIN                     SCALAR      258       1032        4
MAX                     SCALAR      257       1028        4
CMP_IMM                 SCALAR      257       1028        4
LD_XD_XN_IMM            SCALARLDST    2        948      474
MOV_SRC_TO_DST_ALIGNv2  MTE3          1        501      501
DC_PRELOAD_XN_IMM       SCALAR        1        490      490

--- Worst Single Events (top 6 distinct, deduped) ---
LD_XD_XN@SCALARLDST  ts=4432  dur=646  (×255 similar, avg ~5)
MOV_SRC_TO_DST_ALIGNv2@MTE3  ts=10259  dur=501
DC_PRELOAD_XN_IMM@SCALAR  ts=3887  dur=490
LDP_XI_XJ_XN@SCALAR  ts=3902  dur=476  (×3 similar, avg ~475)
LD_XD_XN_IMM@SCALARLDST  ts=3907  dur=474  (×1 similar, avg ~474)
VF@PUSHQ  ts=9878  dur=379

--- Control Flow / Instant Events ---
JUMPC×258  JUMP×217  MOV_SPR_XN×2  NOP×2  WAIT_FLAG_SCALAR×1  BAR×1  END×1

```

## Trace comparison

| Metric | Baseline-equivalent | Optimized | Delta |
|---|---:|---:|---:|
| wall_cycles | 6908 | 6886 | -0.3% |
| hardware latency (sub-kernel) | 2.763 us | 2.754 us | -0.009 us |
| x_events | 2791 | 2583 | -7.5% |
| instant control events | 480 | 482 | +0.4% |
| bottleneck | SCALARLDST 5608 cyc | SCALARLDST 5345 cyc | -4.7% |
| MTE3 busy | 1208 | 880 | -27.2% |
| PUSHQ busy | 717 | 387 | -46.0% |

## Hardware verification (`remote_verify`)

Correctness:
- Optimized `direct_small`: PASS, max_abs=0
- Optimized `persistent_cover`: PASS, max_abs=0
- Optimized `target`: PASS, max_abs=0
- Baseline Triton1/Triton2: skipped in profiler because both contain unsupported `cache_modifier=.cg`.

Benchmark table:

| label | PyTorch / ACL (ms) | Baseline Triton1 | Baseline Triton2 | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_small | 0.023241 | inf | inf | 0.049814 |
| persistent_cover | 0.091801 | inf | inf | 5.657415 |
| target | 3.219044 | inf | inf | 628.010320 |

## Interpretation

The optimized Triton kernel fixes correctness/compilation and slightly improves sub-kernel cannsim bottlenecks, but full target hardware latency is much slower than PyTorch/ACL. Because the editable and read-only Triton baselines both fail on the unsupported cache modifier, no finite Triton baseline latency is available for a direct hardware speedup ratio.
