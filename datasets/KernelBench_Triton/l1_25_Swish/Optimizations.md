# Optimizations Applied: Swish

## 1. Two-path elementwise dispatch

**Code snippet**
```python
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    n_programs = _MAX_PROGRAMS
    _swish_persistent_kernel[(n_programs,)](..., n_programs, BLOCK_SIZE=_BLOCK_SIZE)
else:
    _swish_direct_kernel[(n_tiles,)](..., BLOCK_SIZE=_BLOCK_SIZE)
```

**Rationale**
The benchmark input has `4096 * 393216 = 1,610,612,736` elements. The baseline uses one program per 4096-element tile, which would require `393216` programs and exceeds Ascend FFTS `coreDim <= 65535`; the optimized persistent path caps dispatch to `65535` programs and grid-strides over tiles. The direct path is retained for smaller tensors to avoid persistent-loop overhead when dispatch does not overflow.

## 2. Larger contiguous tile size

**Code snippet**
```python
_BLOCK_SIZE = 8192
offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offs < n_elements
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
```

**Rationale**
Swish is an elementwise activation with contiguous input/output. Increasing the tile from 4096 to 8192 halves the tile count while staying within UB headroom for fp32 vector intermediates (`x`, `sigmoid`, `y`, offsets/masks). This reduces full-shape dispatch pressure and improves per-element cannsim-normalized work.

## 3. Alignment/contiguity hints on offsets

**Code snippet**
```python
tl.multiple_of(offs, 16)
tl.max_contiguous(offs, 16)
```

**Rationale**
The kernel operates on contiguous flattened tensors. The hints expose 16-element alignment/contiguity to the Ascend compiler so GM loads/stores can be coalesced more reliably.

## 4. Preserve fp32 Swish math for precision

**Code snippet**
```python
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
xf = x.to(tl.float32)
y = xf * tl.sigmoid(xf)
tl.store(y_ptr + offs, y, mask=mask)
```

**Rationale**
The baseline computes Swish in fp32 before storing to the output dtype. The optimized kernels preserve the same math and only change scheduling/tiling, so fp16/bf16/fp32 behavior remains aligned with the original operator contract.
