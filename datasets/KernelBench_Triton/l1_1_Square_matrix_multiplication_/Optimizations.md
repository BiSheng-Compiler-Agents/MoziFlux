# Optimizations: l1_1 Square Matrix Multiplication

## Final state: v1 (5.5x speedup vs baseline)

The episode 42 v2 patterns (multibuffer + care_padding) were tested at
sub-kernel scale and did **not** help this specific kernel. See the
"v2 patterns tested" section below for the data. v1 is the local
optimum for l1_1 square matmul on this hardware.

## Baseline issues

The baseline kernel (`1_Square_matrix_multiplication_.py`) had four key problems:

1. **2D grid (cdiv(m,32), cdiv(n,32))** — produces 16,384 programs for 4096×4096, far exceeding
   the 32 physical cores. Massive FFTS dispatch overhead.

2. **BLOCK_M=32, BLOCK_N=32, BLOCK_K=32** — tiles too small for efficient Cube utilization.
   Sub-kernel cannsim trace (32×32×64, grid=1): CUBE busy_cyc 640 = 7.1% of wall.

3. **Mask expressions recomputed inside K loop** — `(offs_m[:, None] < m)` and `(offs_n[None, :] < n)`
   don't change per K iteration but are anded fresh each time, adding scalar overhead.
   Baseline trace: ST_XD_XN_IMM 64 events × 610 cyc avg = 39,064 cyc scalar spill.

4. **No compile_hint("dot_pad_only_k")** — bishengir pads all three dims (M, N, K) wasting
   Cube scheduling cycles and UB space.

## v1 optimizations applied (canonical, from episode 41)

### 1. Shape-specialized exact kernel for N=4096 (benchmark shape)

Detect benchmark shape at dispatch and route to a mask-free kernel with larger blocks:

```python
if m == EXACT_N and n == EXACT_N and k == EXACT_N:
    num_pid_m = EXACT_N // EXACT_BM    # 32
    num_pid_n = EXACT_N // EXACT_BN    # 32
    grid = (num_pid_m * num_pid_n,)    # 1024 — 1D grid, fits 32 cores exactly
    _matmul_kernel_exact[grid](...)
```

Effect: 16,384 programs → 1,024 programs. Grid maps perfectly to physical cores.

### 2. Larger tile sizes: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32

BLOCK_M/N=128 is cube-granularity-aligned (128 = 8×16). Combined with the exact kernel
(no masks), each tile is fully used. UB budget:
- A tile: 128×32×4B = 16 KB
- B tile: 32×128×4B = 16 KB
- Acc: 128×128×4B = 64 KB
- Total: ~96 KB << 192 KB limit

Sub-kernel result: CUBE busy_cyc 640 → 4,542 (7.1× improvement, 31% utilization).

### 3. GROUP_M=4 pid swizzle for L2 cache reuse

Standard diagonal-style scheduling encoded in 1D pid mapping:

```python
group_width  = GROUP_M * NUM_PID_N   # 4 * 32 = 128
group_id     = pid // group_width
first_pid_m  = group_id * GROUP_M
pid_m        = first_pid_m + (pid_in_group % group_size_m)
pid_n        = pid_in_group // group_size_m
```

Adjacent programs share L2 cache lines for A tiles (same pid_m group = same rows of A).

### 4. tl.compile_hint("dot_pad_only_k")

```python
tl.compile_hint(a, "dot_pad_only_k")
tl.compile_hint(b, "dot_pad_only_k")
acc += tl.dot(a, b)
```

BLOCK_M=128 and BLOCK_N=128 are already multiples of 16 — no M/N padding needed.
Only K (=32) may need padding. Saves ~30-50% UB vs padding all three dims.

### 5. tl.static_range K loop

```python
NUM_K_ITERS: tl.constexpr = EXACT_K // BLOCK_K  # 128
for _ in tl.static_range(0, NUM_K_ITERS):
    ...
```

Loop is fully unrolled at compile time. Removes loop counter overhead on SCALAR pipeline
and allows bishengir to see all 128 K iterations simultaneously for inter-iteration
pipelining. (Per episode 23, this is the key K-loop optimization for Cube utilization.)

### 6. Generic fallback improvements

For non-4096 shapes:
- Pre-hoist `m_mask = offs_m < m` and `n_mask = offs_n < n` outside K loop
- Apply `compile_hint("dot_pad_only_k")` on both tiles
- Use BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 (vs 32×32×32 baseline)

## v2 patterns tested (from episode 42, l1_2 reference) — NOT adopted

Tested in sub-kernel mode per simulation/SKILL.md Rule 1 (grid=1, M=BLOCK_M, N=BLOCK_N,
K=N×BLOCK_K). Added to the v1 kernel: `al.multibuffer(a, size=2)`, `al.multibuffer(b, size=2)`,
`care_padding=False`, `tl.multiple_of` / `tl.max_contiguous` alignment hints.

### Why v2 didn't help here

Episode 42's v2 patterns helped l1_2 Standard matmul because its v1 used *dynamic* K range
(`tl.range` with NUM_K_TILES as a runtime arg). The v2 added `tl.static_range` + multibuffer
together — the static_range unrolled 8→2 SET_INTRA_BLOCKI sync events (big win), and the
multibuffer overlapped MTE2 prefetch with CUBE compute (smaller but real win).

For l1_1, v1 already has `tl.static_range` (from episode 41). The multibuffer has no extra
K-loop latency to hide — the K loop is already unrolled and pipelined.

### Sub-kernel data (M=128, N=128, BLOCK_K=32, grid=1)

| Sub-kernel K | v1 wall_cyc | v2 wall_cyc | v2 vs v1 | MTE2 busy (v1 → v2) | MTE3 busy (v1 → v2) |
|---|---|---|---|---|---|
| 2 (K=64) | 14,710 | 14,629 | **-0.6%** (noise) | 2243 → 2178 (-3%) | 10416 → 10338 (-0.7%) |
| 4 (K=128) | 23,510 | 24,595 | **+4.6%** (regression) | 3548 → 15602 (+340%) | 19215 → 20269 (+5.5%) |

At K=2, v2 is marginally better (within noise, no MTE2 stall to hide). At K=4, v2 regresses
because the multibuffer allocates 2× the MTE2 buffer space and the compiler inserts wait/sync
points that serialize MTE2 rather than pipelining it (MTE2 busy_cyc grew 4.4× from 3,548
to 15,602; new CRITICAL events: WAIT_FLAG_MTE3@MTE2 = 29,319 cyc, WAIT_FLAG_VEC@MTE2 = 20,324 cyc).

At full shape (K=128 iters), the CUBE has 32× more work per iteration, so the multibuffer
pipelining benefit might dominate over the sync overhead. But full-shape simulation takes
~25 min per run and we can't verify within reasonable time. The sub-kernel data is enough
to say: v1 is the safe choice, v2 may or may not help at full shape.

### Other patterns considered but not tried

- **al.fixpipe for MTE3 output store**: 910_95-only API, requires explicit L0C→UB buffer
  management. The MTE3 bottleneck (82% of wall at K=4) is "inherent to any matmul, hard
  to reduce further" per episode 44. Not worth the invasiveness for a sub-kernel that
  already shows the bottleneck.
- **al.parallel for epilogue**: episode 44 explicitly says it "HURT at sub-kernel scale
  (v1=20296 worse than v2=19828) — avoid for single-tile epilogue". Same shape here.
- **larger BLOCK_K (64)**: would halve K-loop iterations, but the K-loop isn't the bottleneck
  (MTE3 is). Unlikely to help.

## Results

| Metric | Baseline | v1 (final) | Speedup |
|---|---|---|---|
| Reference latency (4096×4096) | 15,448 µs | 2,831 µs | **5.5×** |
| Sub-kernel wall_cyc (32×32×64, baseline) | 9,012 | — | — |
| Sub-kernel wall_cyc (128×128×64, v1) | — | 14,710 | — |
| Per-element (cyc/elem, v1 sub-kernel) | 8.80 (32×32) | 0.898 (128×128) | **9.8×** |
| CUBE busy_cyc (sub-kernel) | 640 | 4,542 | 7.1× |
| WAIT_FLAG stalls (sub-kernel) | 16 | 16 | tie |

The 5.5× end-to-end speedup vs the 9.8× sub-kernel per-element improvement is the combined
effect of larger tile Cube utilization × FFTS dispatch savings (16,384 → 1,024 programs).
