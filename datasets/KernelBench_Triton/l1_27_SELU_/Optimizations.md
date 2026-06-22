# Optimizations Applied: SELU

## 1. Two-path elementwise dispatch with FFTS grid cap

Baseline launches one program per 4096 elements:

```python
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
_selu_kernel[grid](x_contig, y, n_elements, BLOCK_SIZE=4096)
```

The original shape has `4096 * 393216 = 1,610,612,736` elements, which is `393216` baseline tiles and exceeds Ascend FFTS `coreDim <= 65535`. The optimized host uses a direct path for normal tensors and a persistent grid-stride path only when `n_tiles > 65535`:

```python
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _selu_persistent_kernel[(_MAX_PROGRAMS,)](..., n_programs=_MAX_PROGRAMS)
else:
    _selu_direct_kernel[(n_tiles,)](...)
```

Rationale: persistent loops add control-flow overhead on small tensors, so the direct path remains fastest below the grid cap while the original benchmark shape becomes launchable.

## 2. Larger contiguous tile: `BLOCK_SIZE = 8192`

```python
_BLOCK_SIZE = 8192
offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
tl.multiple_of(offsets, 16)
tl.max_contiguous(offsets, BLOCK_SIZE)
```

Rationale: SELU is a streaming elementwise activation. Doubling tile size halves program count and amortizes scalar/FFTS startup cost while remaining within UB for the fp32 intermediates used by `exp`, `min`, and `max`.

## 3. Single masked vector path for all tiles

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
x32 = x.to(tl.float32)
neg = tl.minimum(x32, 0.0)
pos = tl.maximum(x32, 0.0)
out = pos * SCALE + (tl.exp(neg) - 1.0) * (SCALE * ALPHA)
tl.store(y_ptr + offsets, out.to(x.dtype), mask=mask)
```

Rationale: all accesses are masked for Ascend OOB safety, offsets are contiguous for GM DMA merging, and calculation is in fp32 before casting back to the input dtype.
