# Optimizations Applied: HardTanh

## 1. Two-path elementwise dispatch

The input shape is `4096 x 393216 = 1,610,612,736` elements. The baseline launches one program per 4096-element tile, i.e. `393216` programs, which exceeds the Ascend FFTS 1D grid cap of `65535`.

```python
direct_tiles = triton.cdiv(n_elements, _DIRECT_BLOCK_SIZE)
if direct_tiles > _MAX_PROGRAMS:
    _hardtanh_persistent_kernel[(_MAX_PROGRAMS,)](...)
else:
    _hardtanh_direct_kernel[(direct_tiles,)](...)
```

Rationale: preserve the fast direct launch for normal tensors while routing the benchmark-sized tensor to a legal persistent work loop.

## 2. Persistent-grid kernel for oversized tensors

The persistent path caps the launch grid and lets each program process multiple tiles.

```python
n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
for tile_id in range(pid, n_tiles, n_programs):
    base = tile_id.to(tl.int64) * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE).to(tl.int64)
```

Rationale: `n_tiles=196608` with `BLOCK_SIZE=8192`, so `65535` programs legally cover the original shape in roughly three waves. The `int64` base avoids large-offset overflow on the 6 GB fp32 input.

## 3. Direct-path block retained at 4096, persistent block raised to 8192

```python
_DIRECT_BLOCK_SIZE = 4096
_PERSISTENT_BLOCK_SIZE = 8192
```

Rationale: hardware benchmarking showed the 4096 direct path is better for small/direct tensors, while 8192 halves the persistent tile count for the original oversized tensor.

## 4. Masked contiguous vector access with `care_padding=False`

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
y = tl.where(x > max_val, max_val, tl.where(x < min_val, min_val, x))
tl.store(y_ptr + offsets, y, mask=mask)
```

Rationale: offsets are contiguous for DMA-friendly GM access, all loads/stores are masked, and masked padding values are ignored by the store mask. The nested `tl.where` preserves NaN behavior because comparisons against NaN are false.
