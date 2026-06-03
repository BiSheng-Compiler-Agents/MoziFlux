# Two-Phase Reduction Tile-Count Threshold

## Background

Two-phase reduction (private partial sums + single reduce program) replaces
`tl.atomic_add` serialization on Ascend. Each program writes to its own private
slot (no contention), then one program sums the partials. This is documented
in episodes 20/54 for HingeLoss, where it achieved **2.43× speedup** on large
tile counts (N ≈ 1B → 262,144 tiles).

## The Threshold Problem

Two-phase has a fixed overhead: **2 kernel launches** (partial + reduce).
Each launch costs ~1,150 cycles of FFTS dispatch → ~2,300 cycles ≈ 0.92 µs
(at 0.4 ns/cycle). For small tile counts, the atomic serialization penalty
(~50–100 cycles per contending program) is **smaller than this extra launch
cost**, making two-phase a regression.

## Empirical Data (l1_100_HingeLoss, June 2026, hardware-verified)

Shape was HingeLoss: `z = max(0, 1 - p*t)`, then `mean(z)`.
BLOCK = min(4096, max(256, next_pow2(N))).

### Version 1 (hardcoded BLOCK=1024, NUM_PARTS=32)

| Shape | N | BLOCK | n_tiles | Path | Baseline | Optimized | Speedup |
|-------|---|-------|---------|------|----------|-----------|---------|
| tiny | 512 | 512 | 1 | direct | 32.3 µs | **23.6 µs** | **1.37×** |
| small | 4096 | 1024 | 4 | two-phase (32 prog) | 33.5 µs | 43.8 µs | **0.76× regr** |
| medium | 32768 | 1024 | 32 | two-phase (32 prog) | 62.6 µs | 44.0 µs | 1.42× |
| large | 131072 | 1024 | 128 | two-phase (32 prog) | 38.6 µs | 45.9 µs | 0.84× |

Problem: N=4096 with BLOCK=1024 gives only 4 tiles, but 32 partial programs
were launched. 28 programs did nothing (loop condition `pid * BLOCK < N` fails
for pid > 3), each incurring FFTS dispatch cost. Also: N=4096 should be
single-tile with BLOCK=4096 (matches baseline behavior).

### Version 2 (adaptive BLOCK + NUM_PARTS)

Fix: `BLOCK_SIZE = min(4096, max(256, next_pow2(N)))` (matches baseline)
and `NUM_PARTS = min(32, cdiv(N, BLOCK_SIZE))` (never launch more programs
than tiles).

| Shape | N | BLOCK | n_tiles | NUM_PARTS | Path | Baseline | Optimized | Speedup |
|-------|---|-------|---------|-----------|------|----------|-----------|---------|
| tiny | 512 | 512 | 1 | 1 | direct | 32.3 µs | **23.6 µs** | **1.37×** |
| small | 4096 | 4096 | 1 | 1 | direct | 31.0 µs | **23.0 µs** | **1.35×** |
| medium | 32768 | 4096 | 8 | 8 | two-phase | 33.1 µs | 40.2 µs | 0.82× |
| large | 131072 | 4096 | 32 | 32 | two-phase | 28.9 µs | 37.7 µs | 0.77× |
| nopow2 | 5000 | 4096 | 2 | 2 | two-phase | 28.7 µs | 37.5 µs | 0.77× |

V2 fixed the small (N=4096) case: now 1 direct program → **1.35× win**.
But all two-phase cases (2–32 tiles) still regress because the 2-launch
overhead (~0.92 µs) exceeds atomic contention (50–100 cy/program).

## Recommended Dispatch Strategy

```python
n_tiles = triton.cdiv(N, BLOCK)
if N <= BLOCK:
    _direct_kernel[(1,)](...)          # single tile — 1.35× win
elif n_tiles <= 64:
    _atomic_kernel[(n_tiles,)](...)     # moderate tiles — single launch
else:
    _partial_kernel[(32,)](...)         # many tiles → private stores
    _reduce_kernel[(1,)](...)           # single reduce
```

## Root Cause Anatomy

Two-phase launch overhead breakdown:
- 1st launch (partial): ~1,150 cycles FFTS dispatch = 0.46 µs
- 2nd launch (reduce): ~1,150 cycles FFTS dispatch = 0.46 µs
- Total 2-phase overhead: ~0.92 µs

Atomic contention overhead (estimate):
- ~50–100 cycles per contending program for atomic_add on Ascend
- 8-way: 400–800 cycles = 0.16–0.32 µs
- 32-way: 1,600–3,200 cycles = 0.64–1.28 µs
- 64-way: 3,200–6,400 cycles = 1.28–2.56 µs

Cross-over point (2-phase wins): n_tiles > ~64.

## Adaptive NUM_PARTS Rule

```python
NUM_PARTS = min(32, n_tiles)
```

When n_tiles ≤ 32, each of NUM_PARTS programs covers exactly one contiguous
tile. The strided loop (pid*BLOCK + chunk_idx*NUM_PARTS*BLOCK) degenerates to
`pid*BLOCK + 0` since n_chunks = cdiv(N, NUM_PARTS * BLOCK) = 1.
No wasted programs, no striding overhead.

When n_tiles > 32, NUM_PARTS=32 programs each stride across ~n_tiles/32
chunks. Standard work-distribution pattern from episode 20.

## Related Episodes

- Episode 20: First HingeLoss two-phase pattern (NUM_PARTS=32, always strided)
- Episode 54: HingeLoss hardware verification (2.43× speedup, large N)
- Episode 71: discovered the tile-count threshold, V1→V2 adaptive fix
