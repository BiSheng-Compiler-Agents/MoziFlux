# Optimizations

## 1. Two-path clamp/divide epilogue

Baseline launched one program per epilogue tile:

```python
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
_clamp_divide_inplace_kernel[grid](x, n_elements, min_value, divisor, BLOCK_SIZE=BLOCK_SIZE)
```

For the target output shape `(16, 128, 47, 95, 95)`, `n_elements = 868,710,400` and `ceil(n_elements / 8192) = 106,044`, which exceeds Ascend's `coreDim <= 65,535` launch limit. The optimized host keeps the direct fast path for legal sizes and routes only oversized outputs to a persistent kernel:

```python
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _clamp_divide_persistent_kernel[(_MAX_PROGRAMS,)](
        x, n_elements, _MAX_PROGRAMS, min_value, inv_divisor,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
else:
    _clamp_divide_direct_kernel[(n_tiles,)](...)
```

Rationale: the normal path avoids persistent-loop overhead for small/medium shapes, while the target path becomes legal and still processes all tiles in one launch.

## 2. Reciprocal multiply instead of divide

Baseline epilogue:

```python
x = tl.maximum(x, min_value)
x = x / divisor
```

Optimized epilogue:

```python
inv_divisor = 1.0 / float(divisor)
x = tl.maximum(x, min_value) * inv_divisor
```

Rationale: the divisor is a scalar module parameter; computing its reciprocal once on the host replaces vector division with cheaper vector multiply in the Triton epilogue.

## 3. Int64 tile offsets for oversized tensors

```python
offsets = (tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
```

Rationale: the target output is ~3.47 GB for fp32, so the persistent path may address beyond 2 GiB of byte offset. Promoting offsets prevents int32 pointer-offset overflow hazards on oversized elementwise tensors.

## 4. Preserve ACL ConvTranspose3d

```python
y = F.conv_transpose3d(x, weight, bias=bias, stride=stride, padding=padding)
return _launch_clamp_divide_inplace(y, min_value, divisor)
```

Rationale: ConvTranspose3d is a mature ACL-covered primitive. The optimization targets the custom Triton epilogue legality/performance without replacing vendor convolution.
