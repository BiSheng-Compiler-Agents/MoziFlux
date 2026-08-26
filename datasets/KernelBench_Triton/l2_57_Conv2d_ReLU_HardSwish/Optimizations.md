# Optimizations Applied

## 1. Algebraic ReLU + HardSwish simplification

Baseline epilogue computed ReLU first, then used the ReLU result in the HardSwish clamp:

```python
r = tl.maximum(x, 0.0)
y = r * tl.minimum(r + 3.0, 6.0) * (1.0 / 6.0)
```

The optimized kernel uses the equivalent piecewise form for `HardSwish(ReLU(x))`:

```python
y_pos = x * tl.minimum(x + 3.0, 6.0) * (1.0 / 6.0)
y = tl.where(x > 0.0, y_pos, 0.0)
```

Rationale: for `x <= 0` the output is exactly zero; for `0 < x < 3`, the formula is `x * (x + 3) / 6`; for `x >= 3`, it is `x`. This removes the separate vector max and reduces RVECEX work in the cannsim trace.

## 2. Direct + persistent elementwise dispatch

Optimized host dispatch covers the normal direct path and a grid-capped persistent fallback:

```python
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _relu_hswish_persistent_kernel[(_MAX_PROGRAMS,)](
        x, n_elements, _MAX_PROGRAMS, BLOCK_SIZE=_BLOCK_SIZE, num_warps=8, num_stages=2
    )
else:
    _relu_hswish_direct_kernel[(n_tiles,)](
        x, n_elements, BLOCK_SIZE=_BLOCK_SIZE, num_warps=8, num_stages=2
    )
```

Rationale: the default tensor uses the direct path, while very large tensors avoid Ascend FFTS `coreDim > 65535` launch failure. `profile_kernels.py` force-tests the persistent path by temporarily lowering `_MAX_PROGRAMS`.

## 3. Ascend-safe launch staging and padding hint

The optimized kernels always launch with `num_stages=2` and use `care_padding=False` on masked loads:

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
```

Rationale: `num_stages=1` is an Ascend pitfall; the padding hint is safe here because masked padded lanes are not stored and do not feed reductions.
