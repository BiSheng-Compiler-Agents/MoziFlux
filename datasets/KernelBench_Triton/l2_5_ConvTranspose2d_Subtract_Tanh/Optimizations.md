# Optimizations Applied

## 1. Fixed unsupported Triton tanh API

Baseline code uses `tl.tanh`, which does not exist in this triton-ascend environment and fails during JIT cache-key construction:

```python
y = tl.tanh(z)
```

The optimized kernel uses the supported math intrinsic:

```python
from triton.language.math import tanh as tl_tanh
y = tl_tanh(x.to(tl.float32) - b.to(tl.float32)).to(x.dtype)
```

Rationale: this restores compilability while preserving FP32 tanh evaluation and output dtype casting.

## 2. Retiled the epilogue by `(N, C)` plane instead of flat NCHW element offsets

Baseline computes channel per element:

```python
c_idx = (offs // HW) % C
b = tl.load(b_ptr + c_idx, mask=mask, other=0.0)
```

Optimized code computes the channel once per tile and processes a contiguous spatial slice:

```python
hw_tile = tile_id % NUM_HW_TILES
plane = tile_id // NUM_HW_TILES
c_idx = plane % C
offs_hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
offsets = plane.to(tl.int64) * HW + offs_hw.to(tl.int64)
b = tl.load(b_ptr + c_idx)
```

Rationale: removes thousands of per-element scalar div/mod and bias-address operations, turning the epilogue from scalar-bound to mostly vector/memory work.

## 3. Added direct + persistent dispatch for the Ascend grid cap

The default output shape is `32 x 64 x 513 x 513 = 538,466,304` elements. Baseline's flat `BLOCK_SIZE=8192` launch needs `65,731` programs, exceeding the Ascend FFTS grid cap (`65,535`).

```python
num_hw_tiles = triton.cdiv(HW, _BLOCK_HW)
total_tiles = (N * C) * num_hw_tiles
if total_tiles > _MAX_PROGRAMS:
    _bias_sub_tanh_plane_persistent[(_MAX_PROGRAMS,)](...)
else:
    _bias_sub_tanh_plane_direct[(total_tiles,)](...)
```

Rationale: small/medium shapes use the fastest direct path; huge shapes remain legal and are covered by a persistent tile loop.

## 4. Promoted large offsets to int64

```python
offsets = plane.to(tl.int64) * HW + offs_hw.to(tl.int64)
```

Rationale: the default fp32 output is slightly over 2 GiB, so byte addressing can cross the signed-int32 boundary. Int64 offset arithmetic prevents late persistent tiles from addressing the wrong elements.

## 5. Kept ConvTranspose2d on ACL and fused only the custom epilogue

```python
x = F.conv_transpose2d(...)
return _bias_sub_tanh_fused(x, subtract_bias)
```

Rationale: transposed convolution is a standard ACL-covered primitive; the custom Triton opportunity is the subtract+tanh epilogue and its grid legality/scalar overhead.
