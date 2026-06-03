# Performance Report: l1_2 Standard Matrix Multiplication

## Summary

| Kernel | Sub-kernel cyc/elem | vs Baseline | Notes |
|--------|--------------------:|:-----------:|-------|
| Baseline (`_matmul_kernel`, BLOCK=32) | 8.71 | 1.0x | 2D grid, no swizzle |
| Opt v1 (128x128x32 + GROUP_M + masks + dot_pad_only_k) | 1.03 | **8.5x** | dynamic K loop |
| **Opt v2 (this, + multibuffer + static_range)** | **0.95** | **9.2x** | 1.08x vs v1 |

Sub-kernel traces: grid=1, M=BLOCK_M=128, N=BLOCK_N=128, K=2×BLOCK_K=64.
Normalized per output element: wall_cycles / (BLOCK_M × BLOCK_N).
Hardware latency on physical NPU pending (requires NPU profiling run).

---

## Cannsim Trace Analysis

### Baseline Trace (BLOCK=32×32×32, grid=1, M=32, N=32, K=64)

Wall cycles: **8,919** | x_events: 2,039

Pipeline utilization (busy_cyc):
```
FLOWCTRL   5128  ← BOTTLENECK  (8x SET_INTRA_BLOCKI @ 654 cyc avg)
MTE3       5036
PUSHQ      3762
RVECST     3564  (512 ops)
MTE2       3328
VEC        3231
SCALARLDST 3061  (128 ops — address computation)
RVECLD     2040  (512 ops)
SCALAR     1638
CUBE        640  (only 7.2% utilization!)
```

Top stalls:
- `ST_XD_XN_IMM@SCALARLDST`: 64 ops × 593 avg_cyc = 37,947 total — address/stride overhead
- `WAIT_FLAG_MTE2@VEC`: 4 events × 1,234 avg_cyc — MTE2 pipeline stall
- `CUBE`: only 4 ops (4 MMAD), barely active

Root causes:
1. BLOCK=32×32 too small — Cube scheduler overhead per tile outweighs compute
2. 2D grid × tiny tiles → program-per-core dispatch overhead
3. Masks recomputed per K iteration → scalar overhead
4. No double-buffering → MTE2 stalls block CUBE

---

### Opt v1 Trace (BLOCK=128×128×32, GROUP_M=4, mask hoist, dot_pad_only_k)
Sub-kernel: grid=1, M=128, N=128, K=64

Wall cycles: **16,848** | x_events: 3,459

Pipeline utilization (busy_cyc):
```
FLOWCTRL   9707  ← BOTTLENECK  (8x SET_INTRA_BLOCKI @ 1247 cyc avg)
MTE3       9442
PUSHQ      8910
RVECST     8694  (2048 ops)
RVECLD     6816  (2048 ops)
MTE2       5844
CUBE       4540  (26.9% — 7x improvement vs baseline per tile)
SCALARLDST 3891
SCALAR     2908
```

Key improvements vs baseline:
- CUBE utilization 7.2% → 26.9% (larger tiles)
- Normalized 8.71 → 1.03 cyc/elem (8.5x improvement)
- Only 2 MMAD instructions (K=64 / BLOCK_K=32 = 2 K-iters) — expected

Remaining bottleneck: dynamic `tl.range` loop emits 8 SET_INTRA_BLOCKI sync barriers
per tile (cube-vector pipeline sync). High MTE2 stall cycles (5,844) — no prefetch.

---

### Opt v2 Trace (+ al.multibuffer + tl.static_range)
Sub-kernel: grid=1, M=128, N=128, K=64

Wall cycles: **15,536** | x_events: 5,659

Pipeline utilization (busy_cyc):
```
MTE3      10551  ← BOTTLENECK  (output store — inherent to matmul)
PUSHQ      8816
RVECST     8686  (2048 ops)
FLOWCTRL   6829  (-1.4x vs v1)
RVECLD     6816  (2048 ops)
CUBE       4541  (29.2%)
FIXP       3235
SCALARLDST 3162
MTE2       2420  (-2.4x vs v1 — multibuffer working!)
SCALAR     1477
```

Key improvements vs opt v1:
- MTE2 busy_cyc: **5,844 → 2,420** (2.4× reduction) — multibuffer overlapping DMA/CUBE
- FLOWCTRL: **9,707 → 6,829** (1.4× reduction) — static_range reduces sync barriers
- SET_INTRA_BLOCKI count: **8 → 2** (4× fewer cube-vector sync points)
- Wall_cycles: **16,848 → 15,536** (-8%), normalized 1.03 → 0.95 cyc/elem (1.08×)

New bottleneck: MTE3 (C tile write to GM) at 10,551 cyc = 68% of wall.
This is the output store and is inherent to any matmul — hard to reduce further.
WAIT_FLAG_VEC@MTE3 total: 15,951 cyc (up from 9,951 in v1 — the store pipeline is now
the dominant limiter).

---

## Comparison Table

| Metric                  | Baseline | Opt v1  | Opt v2  | v2 vs baseline |
|-------------------------|----------|---------|---------|----------------|
| wall_cycles (sub-kern)  | 8,919    | 16,848  | 15,536  | —              |
| cyc/output elem         | 8.71     | 1.03    | 0.95    | **9.2×**       |
| CUBE utilization        | 7.2%     | 26.9%   | 29.2%   | 4× better      |
| MTE2 busy_cyc           | 3,328    | 5,844   | 2,420   | 1.4× better    |
| FLOWCTRL busy_cyc       | 5,128    | 9,707   | 6,829   | 1.3× better    |
| SET_INTRA_BLOCKI count  | 8        | 8       | 2       | 4× fewer       |
| Bottleneck pipeline     | FLOWCTRL | FLOWCTRL| MTE3    | shifted to store|

---

## Hardware Latency (NPU profiling — to be updated after profiling run)

Shape: M=1024, K=4096, N=2048 (benchmark shape)

| Kernel       | Latency (µs) | Speedup |
|--------------|:------------:|:-------:|
| torch.matmul | TBD          | ref     |
| Baseline     | TBD          | 1.0×    |
| Optimized    | TBD          | ~9×     |

_Run `python profile_kernels.py --bench` on physical NPU to populate this table._
