# Optimizations: HardSigmoid

## 1. Two-path elementwise dispatch

The input shape is `4096 x 393216` (`1,610,612,736` elements). The baseline launches one program per 2048-element tile, so the original shape needs `786432` programs and exceeds the Ascend FFTS `coreDim <= 65535` cap.

```python
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _hardsigmoid_persistent_kernel[(_MAX_PROGRAMS,)](..., _MAX_PROGRAMS,
        BLOCK_SIZE=_BLOCK_SIZE)
else:
    _hardsigmoid_direct_kernel[(n_tiles,)](..., BLOCK_SIZE=_BLOCK_SIZE)
```

Rationale: direct launch is kept for normal sizes to avoid persistent-loop overhead; the persistent path is used only when the tile count would overflow the hardware launch cap.

## 2. Larger contiguous tile size

The optimized kernel increases the tile from 2048 to 8192 elements.

```python
_BLOCK_SIZE = 8192
offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
tl.multiple_of(offsets, 16)
tl.max_contiguous(offsets, BLOCK_SIZE)
```

Rationale: HardSigmoid is a memory-light elementwise activation, so larger contiguous tiles amortize per-program scalar/FFTS overhead while staying within UB capacity for one input tile plus output expression temporaries.

## 3. Separate persistent kernel with tile-stride loop

```python
n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
for tile_id in range(pid, n_tiles, n_programs):
    offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

Rationale: the loop iterates over tile IDs, not raw elements, so every large-tensor tile is covered exactly once with a capped grid.

## 4. Precision and branch semantics preserved

```python
x32 = x.to(tl.float32)
y_mid = x32 * (1.0 / 6.0) + 0.5
y32 = tl.where(x32 <= -3.0, 0.0, y_mid)
y32 = tl.where(x32 >= 3.0, 1.0, y32)
tl.store(y_ptr + offsets, y32.to(x.dtype), mask=mask)
```

Rationale: the piecewise `tl.where` form matches the baseline HardSigmoid semantics, including NaN propagation through the middle expression, while computing the affine expression in fp32 before casting back to the input dtype.
