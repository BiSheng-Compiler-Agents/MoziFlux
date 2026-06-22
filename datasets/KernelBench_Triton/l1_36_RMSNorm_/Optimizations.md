# Optimizations for 36_RMSNorm_

## 1. Removed NHWC materialization round trips

Baseline converts NCHW input to NHWC, runs a row-wise RMSNorm, then permutes back:

```python
x_nhwc = x.permute(0, 2, 3, 1).contiguous()
x_2d = x_nhwc.view(B * H * W, C)
...
return y_2d.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
```

Optimized kernel keeps the original contiguous NCHW allocation and normalizes across channels for a block of spatial positions:

```python
x_3d = x.view(B, C, H * W)
y_3d = torch.empty_like(x_3d)
_rmsnorm_nchw_hw_kernel[grid](x_3d, y_3d, B, hw, C, ...)
return y_3d.view(B, C, H, W)
```

Rationale: eliminating two full-tensor layout conversions removes the dominant GM traffic at the target shape.

## 2. Spatial blocking: one program handles 128 HW positions

```python
offs_hw = tile_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
sumsq = tl.zeros([BLOCK_HW], dtype=tl.float32)
...
sumsq += tl.sum(x_f32 * x_f32, axis=0)
```

Rationale: the baseline maps one program to one `(B,H,W)` row of 64 channels. The optimized program handles 128 rows at once, amortizing scalar launch/control overhead and using vector lanes across spatial positions.

## 3. Persistent 1D grid cap for target-sized tensors

```python
total_tiles = B * triton.cdiv(hw, block_hw)
grid = (min(total_tiles, _MAX_PROGRAMS),)
...
for tile in tl.range(pid, total_tiles, nprog):
    ...
```

Rationale: the target shape has `112 * ceil(512*512/128) = 229376` spatial tiles, exceeding Ascend `coreDim <= 65535`. The persistent loop preserves correctness and avoids runtime launch failure.

## 4. FP32 reduction with fully masked memory operations

```python
x = tl.load(ptrs, mask=mask_c[:, None] & mask_hw[None, :], other=0.0)
x_f32 = x.to(tl.float32)
sumsq += tl.sum(x_f32 * x_f32, axis=0)
tl.store(y_ptrs, y, mask=mask_c[:, None] & mask_hw[None, :])
```

Rationale: RMS accumulation stays in FP32 for numerical parity with PyTorch/NPU; all loads and stores are masked for channel and HW boundaries.
