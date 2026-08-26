# Optimizations Applied

## 1. Production dispatch to CANN/ACL standard operators

**Code**
```python
return self.avgpool(torch.tanh(x - self.subtract1_value)) - self.subtract2_value
```

**Rationale**
The input baseline already uses `nn.Conv2d` for the convolution, then launches a custom Triton epilogue for `subtract -> tanh -> avgpool -> subtract`. Cannsim micro-tracing showed that this custom epilogue is dominated by scalar/indexing work (`SCALARLDST + SCALAR + PUSHQ`), which matches the standard Conv2d + activation + pool dispatch pattern: keep convolution/pooling/activation on ACL unless a custom epilogue trace clearly wins.

## 2. Grid-cap-safe Triton fallback for the fused epilogue

**Code**
```python
n_tiles = N * C * triton.cdiv(outH * outW, _BLOCK_HW)
if n_tiles > _MAX_GRID:
    n_programs = _MAX_GRID
    _tanh_avgpool_persistent_kernel[(n_programs,)](..., n_tiles, n_programs, ...)
else:
    _tanh_avgpool_direct_kernel[(n_tiles,)](..., n_tiles, ...)
```

**Rationale**
The default post-convolution shape is `N*C*ceil(outH*outW/1024)=65536` epilogue tiles, exactly over the Ascend FFTS grid cap. The optimized file keeps a direct path for small/irregular shapes and a persistent path for grid-cap shapes; `profile_kernels.py` explicitly unit-tests both fallback dispatch paths.

## 3. Safe masked pointer derivation for padded lanes

**Code**
```python
mask_hw = (tile_id < n_tiles) & (offs_hw < (outH * outW))
safe_hw = tl.where(mask_hw, offs_hw, 0)
oh = safe_hw // outW
ow = safe_hw - oh * outW
```

**Rationale**
Ascend requires that masked lanes do not form invalid addresses before the mask is applied. The fallback remaps inactive lanes to element zero before computing `oh/ow` and strided pointers.

## 4. Fast bounded tanh in the Triton fallback

**Code**
```python
z = tl.minimum(tl.maximum(2.0 * x, -20.0), 20.0)
return 2.0 / (1.0 + tl.exp(-z)) - 1.0
```

**Rationale**
This avoids the baseline fallback's `abs/sign/(1-e)/(1+e)` sequence. The remote unit test verified both direct and persistent fallback paths against the PyTorch/ACL reference with max absolute error within tolerance.
