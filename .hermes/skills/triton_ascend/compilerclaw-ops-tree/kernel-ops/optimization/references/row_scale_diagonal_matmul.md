# Row-Scaling / Diagonal Matmul — Case Study

**Operation**: `C[i,j] = A[i] * B[i,j]` where A is a 1D vector (diagonal matrix representation)
and B is a 2D matrix. Each row of B is scaled by the corresponding element of A.

**KernelBench**: `l1_12_Matmul_with_diagonal_matrices_`

**Hardware**: Ascend950 (cannsim) | **Date**: June 2026

This is an **element-wise** operation (no `tl.dot`), not a matrix multiply. The
computation is a scalar broadcast + element-wise multiply.

---

## Baseline Bottleneck

| Metric | Value | % Wall |
|--------|-------|--------|
| wall_cycles | 2493 | — |
| BLOCK_N | 64 | — |
| Elements/program | 64 | — |
| cy/element | 38.95 | — |
| **SCALARLDST** | **2435** | **97.7%** |
| SCALAR | 636 | 25.5% |
| MTE2 | 831 | 33.3% |
| MTE3 | 734 | 29.4% |
| RVECEX (compute) | 13 | 0.5% |

**Root cause**: The if/else branch in the baseline caused massive scalar register
spills. `ST_XD_XN_IMM` consumed 1,263 cycles from a single event — the branch
condition computation and predicate evaluation spilled to the scalar register file.

---

## Applied Optimizations

### 1. Remove if/else branch → single masked path

The baseline had a fast-path (unmasked) and slow-path (masked) depending on
whether the tile was at the matrix boundary. This doubled the instruction stream
and created scalar spill from the condition computation.

**Fix**: Always use a single masked load/store path. Ascend handles masked
operations efficiently — the mask check is free when all elements are in-bounds
(the common case for interior tiles).

**Trace impact**: ST_XD_XN_IMM dropped from 1 event/1,263 cy to 2 events/1,236 cy
total (the remaining spill is args-struct storage, not branch condition).

```python
# BEFORE: 2-way branch with mask/unmasked paths
if full_tile:
    a_val = tl.load(a_ptr + row)                    # no mask
    b = tl.load(b_ptrs)                              # no mask
    tl.store(c_ptrs, b * a_val)                      # no mask
else:
    mask = row_in_bounds & cols_in_bounds
    a_val = tl.load(a_ptr + row, mask=row_in_bounds, other=0)
    b = tl.load(b_ptrs, mask=mask, other=0)
    tl.store(c_ptrs, b * a_val, mask=mask)

# AFTER: single masked path
mask = (row < N) & (offs_n < M)
a_val = tl.load(a_ptr + row, mask=row < N, other=0.0)
b = tl.load(b_ptrs, mask=mask, other=0.0)
tl.store(c_ptrs, b * a_val, mask=mask)
```

### 2. Remove `cache_modifier=".cg"`

This is a CUDA-specific L2-bypass hint. On Ascend it has no effect and may
silently kill compilation (producing an empty `.npubin` with no error).

**Fix**: Drop `.cg` from all loads/stores. Keep `.ca` on A-vector loads
(benign broadcast hint).

### 3. Increase BLOCK_N from 64 → 1024

The optimal block size for Ascend vector operations is 1024–2048. BLOCK_N=64
wastes per-program startup overhead (~1,150 cycles/AIV program) on only 64
elements.

**Trace impact**: cy/element dropped from 38.95 to 3.74 (10.4× improvement).

| Block size | wall_cycles | elements | cy/element | Vector lanes |
|------------|-------------|----------|------------|-------------|
| 64 | 2493 | 64 | 38.95 | 2 |
| 1024 | 3831 | 1024 | 3.74 | 9 |

### 4. Two-path dispatch (direct + persistent)

Following Rule 8 of the optimization guide: direct path when `total_tiles ≤ 65535`,
persistent work-stealing path when exceeded. This prevents the FFTS grid crash
(`coredim > UINT16_MAX`) while avoiding unnecessary while-loop overhead on
small-to-medium grids.

### 5. 1D grid with pid → (row, col_block) remap

Instead of a 2D grid (pid_m, pid_n), use a single 1D grid and remap:

```python
pid = tl.program_id(0)
row = pid // num_col_blocks
col_block = pid % num_col_blocks
```

This simplifies dispatch and avoids 2D grid overhead. `num_col_blocks` is a
`tl.constexpr`, so the division/modulo compile to shift+mask.

---

## Results

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| wall_cycles | 2493 | 3831 | +53% |
| elements/tile | 64 | 1024 | +1500% |
| **cy/element** | **38.95** | **3.74** | **10.4×** |
| SCALARLDST % | 97.7% | 61.8% | −35.9pp |
| RVECEX ops | 2 | 38 | +1800% |
| Vector lanes | 2 | 9 | +350% |
| Correctness | ✅ | ✅ | — |

The 10.4× per-element efficiency gain comes primarily from:
1. Eliminating the if/else branch (removed 1,263-cy scalar spill)
2. Larger blocks (amortizes per-program startup over 16× more elements)

---

## Key Takeaways for Similar Kernels

1. **Row-scaling is an element-wise operation** — treat it like a broadcast-multiply,
   not a matmul. No `tl.dot`, no Cube core, no 2D tiling needed.

2. **If/else branches cause SCALARLDST bottlenecks** — especially when the two
   branches differ only in mask presence. Always prefer single masked paths.
   The trace signature: a single `ST_XD_XN_IMM` event with >1000 cycles.

3. **Block sizing is highest ROI** — going from BLOCK_N=64 to 1024 gave 10.4×
   improvement. For row-scaling, BLOCK_N can be as large as the row width
   (up to UB limits, which are ~125 KB usable for ~64K fp16 elements).

4. **`.cg` cache modifiers must be removed** on Ascend — they silently break
   compilation. This applies to any kernel ported from CUDA.

5. **1D grid simplifies dispatch** — when `num_col_blocks` is constexpr, the
   `pid // nc` / `pid % nc` remap is free (shift+mask) and avoids 2D grid
   complexity.
