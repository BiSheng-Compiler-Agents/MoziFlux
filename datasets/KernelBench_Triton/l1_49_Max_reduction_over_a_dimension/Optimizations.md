# Optimizations Applied

## 1. Vectorized `dim=1` reduction tile

Baseline streamed one M row at a time inside a `while` loop:

```python
for mi in tl.static_range(0, BLOCK_M):
    row = tl.load(base + m_idx * sx1 + n_offsets * sx2, mask=...)
    acc = tl.maximum(acc, row)
```

Optimized code loads a `[BLOCK_M, BLOCK_N]` tile and reduces it with `tl.max(axis=0)`:

```python
vals = tl.load(ptrs, mask=mask, other=-float("inf"), cache_modifier=".cg").to(tl.float32)
tile_max = tl.max(vals, axis=0)
acc = tl.maximum(acc, tile_max)
```

Rationale: the target shape reduces 4096 rows for each `(B, N-block)` tile. Raising `BLOCK_M` from 8 scalar row updates to a 32-row SIMD reduction cuts loop/control work and lets the vector unit perform the local max reduction.

## 2. 1D persistent dispatch for all reduction dimensions

Baseline used 2D logical grids such as `(B, cdiv(N, BLOCK_N))` and `(M, cdiv(N, BLOCK_N))`. The optimized kernels flatten logical tiles and launch a capped 1D grid:

```python
total_tiles = B * triton.cdiv(N, BLOCK_N)
n_programs = min(total_tiles, vec_cores, 65535)
_kernel[(n_programs,)](..., total_tiles, n_programs, BLOCK_M=32, BLOCK_N=128)
```

Device code then processes tiles with a grid-stride loop:

```python
for tile_id in tl.range(pid, total_tiles, n_programs):
    ...
```

Rationale: this avoids Ascend `coreDim > 65535` hazards on non-default dimensions and reduces launch scheduling overhead for the default target.

## 3. FP32 reduction accumulator and masked boundary loads

Every reduction path uses FP32 accumulation and masks all loads/stores:

```python
acc = tl.full([BLOCK_N], -float("inf"), tl.float32)
vals = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
tl.store(out_ptrs, acc.to(out.dtype.element_ty), mask=out_mask)
```

Rationale: FP16/BF16 max reductions remain numerically stable and Ascend out-of-bounds accesses are prevented for non-power-of-two dimensions such as `N=4095`.
