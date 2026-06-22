# Optimizations Applied

## 1. Remove the custom Triton epilogue from the production path

Baseline performed ACL `F.conv2d` and then launched `_scale_min_channel_kernel` to compute `min(conv_out * scale, dim=1)`. The optimized default path keeps the mature ACL convolution and replaces the custom strided-channel Triton reduction with ACL reductions:

```python
if self.scale_factor >= 0.0:
    return torch.amin(x, dim=1, keepdim=True).mul(self.scale_factor)
return torch.amax(x, dim=1, keepdim=True).mul(self.scale_factor)
```

Rationale: `scale_factor` is a host scalar, so `min(scale*x)` is `scale*min(x)` for non-negative scale and `scale*max(x)` for negative scale. This removes the custom Triton launch, avoids strided channel-plane gathers in UB, and preserves semantics for negative scale without adding baseline-incompatible guards.

## 2. Preserve a tested Triton fallback with direct + persistent dispatch

A `force_triton=True` diagnostic path remains for dispatch coverage and cannsim comparison:

```python
n_tiles = B * triton.cdiv(H * W, _BLOCK_HW)
if n_tiles > _MAX_PROGRAMS:
    _scale_min_channel_persistent_kernel[(_MAX_PROGRAMS,)](...)
else:
    _scale_min_channel_direct_kernel[(n_tiles,)](...)
```

Rationale: the production optimization is ACL dispatch, but the fallback keeps a legal custom path for small direct grids and oversized grid-capped cases. The persistent kernel iterates over tiles, not elements, and uses int64 HW offsets for the oversized path.

## 3. Larger HW tile for fallback reduction

The fallback maps one program to one `(batch, HW tile)` and reduces channel chunks in UB:

```python
vals = tl.load(x_ptr + b_idx * stride_xn + c_idx[:, None] * stride_xc + h[None, :] * stride_xh + w[None, :] * stride_xw,
               mask=mask, other=0.0).to(tl.float32)
acc = tl.minimum(acc, tl.min(vals * scale, axis=0).to(tl.float32))
```

Rationale: this avoids the baseline's extra `BLOCK_B` dimension and supports a larger contiguous HW tile in production (`_BLOCK_HW=256`) while retaining fp32 reduction precision and complete masks.
