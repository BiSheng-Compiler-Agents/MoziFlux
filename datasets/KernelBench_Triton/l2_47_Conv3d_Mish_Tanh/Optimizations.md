# Optimizations Applied

## 1. Stable one-exp `tanh(softplus(x))` for Mish

Baseline computes `tanh(softplus(x))` through `exp(-abs(x))` plus `exp(2 * max(x, 0))`, which adds a second vector exponential and can overflow for large positive inputs:

```python
e_neg_ax = tl.exp(-ax)
e2max = tl.exp(2.0 * tl.maximum(x_f32, 0.0))
e2s = e2max * one_plus * one_plus
tanh_sp = 1.0 - 2.0 / (1.0 + e2s)
```

The optimized kernel uses the algebraically equivalent ratio with `z = exp(-abs(x))`:

```python
z = tl.exp(-tl.abs(x_f32))
z2 = z * z
pos = (1.0 + 2.0 * z) / (1.0 + 2.0 * z + 2.0 * z2)
neg = (z2 + 2.0 * z) / (z2 + 2.0 * z + 2.0)
tanh_sp = tl.where(x_f32 >= 0.0, pos, neg)
```

Rationale: this removes one `RV_VEXP` group and avoids `exp(2*x)` overflow while keeping fp32 activation math.

## 2. Hardware tanh for the final activation

Baseline expands the final `tanh(mish)` into `abs`, `exp`, reciprocal/division, and sign handling:

```python
am = tl.abs(mish)
e2a = tl.exp(2.0 * am)
sign = tl.where(mish >= 0.0, 1.0, -1.0)
out_f32 = sign * (1.0 - 2.0 / (1.0 + e2a))
```

The optimized kernel calls Ascend Triton math tanh directly:

```python
from triton.language.math import tanh as tl_tanh
out = tl_tanh(mish).to(x.dtype)
```

Rationale: `tl_tanh` maps to the vector math implementation and cuts the manual instruction sequence. Cannsim showed `RVECEX` busy cycles drop from 2013 to 1340.

## 3. Grid-cap-safe direct + persistent dispatch

The default output has 28,830 epilogue tiles at `BLOCK_SIZE=4096`, so direct dispatch is fastest. Larger legal tensors can exceed Ascend's 65,535 grid cap; the optimized host has a persistent fallback.

```python
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_GRID:
    _mish_tanh_persistent_kernel[(_MAX_GRID,)](..., n_programs=_MAX_GRID)
else:
    _mish_tanh_direct_kernel[(n_tiles,)](...)
```

Rationale: preserves direct-launch performance for normal shapes and correctness for large tensors without adding new input restrictions. `profile_kernels.py` includes a forced-persistent unit test.

## 4. Contiguous epilogue input and padding hints

The epilogue keeps the post-convolution tensor contiguous and uses one-dimensional contiguous offsets:

```python
x_contig = x.contiguous()
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
tl.max_contiguous(offs, BLOCK_SIZE)
x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
```

Rationale: the activation is pure elementwise; contiguous tiling maximizes DMA coalescing and `care_padding=False` avoids extra padding work on masked tail lanes.
