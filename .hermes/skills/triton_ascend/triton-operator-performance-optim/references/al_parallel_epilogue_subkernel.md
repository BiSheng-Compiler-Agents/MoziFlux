# al.parallel for GEMM Epilogue — Sub-Kernel Scale Finding

## Result (l2_9 Matmul_Subtract_Multiply_ReLU, June 2026)

Tested three variants with cannsim sub-kernel host (M=N=128, K=64, grid=1):

| Variant | wall_cycles | vs baseline |
|---------|-------------|-------------|
| Baseline (dynamic range, no optimizations) | 20397 | — |
| Opt v1: all opts + al.parallel epilogue | 20296 | -0.5% |
| Opt v2: all opts, NO al.parallel | **19828** | **-2.8%** |

**al.parallel HURT at sub-kernel tile scale.**

## Why It Hurts

Epilogue: `acc + bias → subtract → multiply → ReLU` — only 3 arithmetic ops on a 128×128 fp32 tile.

With al.parallel:
- SET_INTRA_BLOCKI avg cyc: 1118 → 4704 (+4.2×)
- WAIT_FLAG_VEC@MTE3 avg cyc: 3224 → 5641 (+1.75×)

The parallel block coordination (SET_INTRA_BLOCKI sync between cube and vector subcores) adds more overhead than the 2× compute parallelism saves for a trivial epilogue.

## Rule

Use `al.parallel(bind_sub_block=True)` for GEMM epilogues ONLY when:
1. Post-dot computation per tile is substantial (e.g., layer norm, softmax, multi-step reduction)
2. Tile is large enough that per-tile sync overhead (~4000 cyc) is < 10% of epilogue compute

For simple fused activations (bias + activate), skip al.parallel entirely.

## Trace Comparison

### v1 (with al.parallel) top stalls:
```
SET_INTRA_BLOCKI  FLOWCTRL  4 events  18815 total_cyc  avg 4704 cyc
WAIT_FLAG_VEC     MTE3      4 events  22565 total_cyc  avg 5641 cyc
```

### v2 (no al.parallel) top stalls:
```
SET_INTRA_BLOCKI  FLOWCTRL  4 events  17403 total_cyc  avg 4351 cyc
WAIT_FLAG_VEC     MTE3      3 events  19289 total_cyc  avg 6430 cyc
```

v2 has 1 fewer WAIT_FLAG_VEC@MTE3 event — the output write completes faster without the parallel sync overhead.

## MTE3 as Sub-Kernel Bottleneck

At sub-kernel scale (grid=1, one output tile), MTE3 (UBUF→GM write of BLOCK_M×BLOCK_N output)
always appears as the bottleneck. For 128×128 fp32 that is 64KB of output writes.

**This is inherent — do not try to reduce MTE3 in sub-kernel traces.**

Full-shape gains from GROUP_M swizzle (L2 reuse) are invisible at sub-kernel scale.
Focus sub-kernel optimization on: scalar spill (ST_XD_XN_IMM), K-loop sync (SET_INTRA_BLOCKI),
and CUBE utilization (MMAD count / wall_cycles ratio).
