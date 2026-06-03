# Performance Report: l1_1 Square Matrix Multiplication

## Summary

| Kernel | Reference latency (4096×4096) | Sub-kernel wall_cyc (K=2) | Sub-kernel wall_cyc (K=4) | Per-element (cyc/elem) |
|---|---|---|---|---|
| Baseline | 15,448 µs | 9,012 | — | 8.80 (32×32) |
| v1 (final) | 2,831 µs | 14,710 | 23,510 | 0.898 (128×128) |
| v2 (tested, not adopted) | — | 14,629 (-0.6%) | 24,595 (+4.6%) | 0.893 |

v1 is the final adopted kernel. v2 patterns from episode 42 were tested at sub-kernel scale
and did not help this specific kernel (see Optimizations.md for the analysis).

## Method (sub-kernel, simulation/SKILL.md Rule 1)

All cannsim traces use the sub-kernel pattern: `grid=(1,1,1)`, M=BLOCK_M, N=BLOCK_N,
K=N×BLOCK_K. The K-loop iteration count (N) is varied to verify the effect of pipeline depth.

| Sub-kernel | M | N | K | BLOCK_M | BLOCK_N | BLOCK_K | grid |
|---|---|---|---|---|---|---|---|
| Baseline | 32 | 32 | 64 | 32 | 32 | 32 | 1×1×1 |
| v1 K=2 | 128 | 128 | 64 | 128 | 128 | 32 | 1×1×1 |
| v1 K=4 | 128 | 128 | 128 | 128 | 128 | 32 | 1×1×1 |
| v2 K=2 | 128 | 128 | 64 | 128 | 128 | 32 | 1×1×1 |
| v2 K=4 | 128 | 128 | 128 | 128 | 128 | 32 | 1×1×1 |

Trace files: `/tmp/cannsim_matmul_baseline_sub_*.json`, `/tmp/cannsim_matmul_opt_sub_*.json`,
`/tmp/cannsim_matmul_opt_v2_*.json`. Aggregated summaries in
`/tmp/trace_summary_baseline_sub.txt`, `/tmp/trace_summary_opt_v1.txt`,
`/tmp/trace_summary_opt_v2.txt`, `/tmp/trace_summary_opt_v1_k4.txt`,
`/tmp/trace_summary_opt_v2_k4.txt`.

## Baseline Sub-Kernel Trace (32×32×64, grid=1)

Wall cycles: 9,012 | Events: 2,039

Pipeline utilization:
- FLOWCTRL: 5,231 busy_cyc ← BOTTLENECK (58%)
- MTE3:    5,139 busy_cyc
- PUSHQ:   3,762 busy_cyc
- MTE2:    3,498 busy_cyc
- VEC:     3,401 busy_cyc
- CUBE:      640 busy_cyc (7.1% utilization — severely underutilized)

Top stalls:
- ST_XD_XN_IMM × 64 events @ 610 cyc avg = 39,064 cyc scalar spill
- WAIT_FLAG_VEC@MTE3 × 4 @ 2,035 cyc avg
- SET_INTRA_BLOCKI × 8 @ 667 cyc avg

Root causes:
1. 32×32 tiles give poor Cube utilization (7.1%)
2. Masks recomputed each K iteration → ST_XD_XN_IMM scalar spill
3. No dot_pad_only_k → M/N padding waste

## v1 Sub-Kernel Trace (128×128×64, K=2, grid=1)

Wall cycles: 14,710 | Events: 5,278

Pipeline utilization:
- MTE3:   10,416 busy_cyc ← BOTTLENECK (71% — output store to GM)
- PUSHQ:   8,816 busy_cyc
- RVECST:  8,686 busy_cyc
- RVECLD:  6,816 busy_cyc
- FLOWCTRL: 6,693 busy_cyc
- CUBE:    4,542 busy_cyc (31% utilization)

Key improvements vs baseline:
- CUBE busy_cyc: 640 → 4,542 (7.1× improvement)
- Per-element: 8.80 → 0.898 cyc/elem (9.8× faster)
- MMAD instructions confirmed present
- No more ST_XD_XN_IMM scalar spill (hoisted masks work)
- SET_INTRA_BLOCKI count: 8 → 2 events (static_range working)

## v1 Sub-Kernel Trace (128×128×128, K=4, grid=1)

Wall cycles: 23,510 | Events: 10,375

Pipeline utilization:
- MTE3:   19,215 busy_cyc ← BOTTLENECK (82% — output store to GM)
- PUSHQ:  17,603 busy_cyc
- RVECST: 17,364 busy_cyc
- FLOWCTRL: 15,493 busy_cyc
- RVECLD: 13,632 busy_cyc
- CUBE:    9,083 busy_cyc (39% utilization)

K=4 SET_INTRA_BLOCKI: 4 events @ 8,257 cyc avg = 33,028 cyc. MTE3 dominates wall.

## v2 Sub-Kernel Trace (128×128×64, K=2, grid=1) — TESTED BUT NOT ADOPTED

Wall cycles: 14,629 (-0.6% vs v1 14,710 — within noise)

Differences from v1:
- MTE2 busy: 2243 → 2178 (-3%)
- WAIT_FLAG_MTE2@VEC: 3453 → 3310 (-4%)
- Event counts IDENTICAL to v1 (5278 events, 60 unique names)
  → multibuffer is purely a scheduling hint, no new instructions

The 0.6% wall difference is within noise; the K=2 sub-kernel is too short for the
multibuffer to provide meaningful pipelining benefit. MTE2 wasn't the bottleneck
(only 2243 busy_cyc / 15% of wall), so there's nothing to hide.

## v2 Sub-Kernel Trace (128×128×128, K=4, grid=1) — TESTED BUT NOT ADOPTED

Wall cycles: 24,595 (+4.6% vs v1 23,510 — REGRESSION)

The K=4 result shows the multibuffer actually *hurts* at this sub-kernel size:
- MTE2 busy: 3548 → 15602 (+340%, 4.4× more)
- WAIT_FLAG_MTE3@MTE2: NEW CRITICAL event, 29,319 cyc total (×2 events)
- WAIT_FLAG_VEC@MTE2: NEW CRITICAL event, 20,324 cyc total (×2 events)
- WAIT_FLAG_MTE2@VEC: 9435 → 25573 (+171%)

The multibuffer allocates 2× the MTE2 buffer space; the compiler inserts wait/sync
points that serialize MTE2 rather than pipelining it. CUBE only has 9,083 busy_cyc
to consume the prefetched MTE2, so MTE2 stalls waiting for CUBE more than it overlaps.

This K=4 regression would likely disappear at K=128 (full shape) where CUBE has
32× more work, but we can't verify within reasonable simulation time. The sub-kernel
data is enough to say: **v1 is the safe choice for this kernel**.

## Per-element normalization

Comparing kernels at the same per-output-element cost:

| Kernel | wall_cyc | output elems | cyc/elem |
|---|---|---|---|
| Baseline (32×32) | 9,012 | 1,024 | **8.80** |
| v1 (128×128) | 14,710 | 16,384 | **0.898** |
| v2 (128×128) | 14,629 | 16,384 | **0.893** |

Speedup:
- v1 vs baseline: 8.80 / 0.898 = **9.8× per output element**
- v2 vs baseline: 8.80 / 0.893 = **9.85× per output element**
- v2 vs v1: 0.898 / 0.893 = **1.006× (within noise)**

## Reference latency (full 4096×4096, from episode 41)

The 5.5× end-to-end speedup (15,448 µs → 2,831 µs) on 4096×4096 is the combined
effect of:
1. Per-tile instruction mix improvement (sub-kernel: 9.8× per element)
2. FFTS dispatch savings from 16,384 → 1,024 programs (≈ 16× fewer program launches)
3. GROUP_M=4 swizzle for L2 cache reuse across the 1,024-program grid

FFTS dispatch savings are invisible at sub-kernel scale (only 1 program) — they require
a full-shape hardware run to measure. Per simulation/SKILL.md Rule 1, we use sub-kernel
to verify the per-tile mix; the dispatch savings are inherited from the v1 design.
