# Optimizations Applied

## 1. Keep Conv3d on ACL and optimize only the post-convolution epilogue

```python
x = self.conv(x)
return _softmax_then_two_pools_fused_triton(x, self.pool_kernel_size)
```

The convolution is a standard `nn.Conv3d` and should stay on CANN/ACL. The custom Triton work is limited to the `softmax(dim=1) -> MaxPool3d -> MaxPool3d` epilogue, where fusion avoids materializing the full softmax tensor before pooling.

## 2. Channel-last materialization for the fused softmax/pool window

```python
x_last = x.permute(0, 2, 3, 4, 1).contiguous()
y_last = torch.empty((N, OD, OH, OW, C), device=x.device, dtype=x.dtype)
```

The baseline softmax reads channel values from contiguous NCDHW with a large `D*H*W` stride between channels. Materializing NDHWC makes the channel reduction contiguous for each spatial point, reducing scalar address work and scalar load/store pressure in the traced kernel.

## 3. Direct + persistent grid-capped dispatch

```python
total_tiles = N * OD * OH * tiles_ow
if total_tiles <= _MAX_GRID:
    _softmax_pool2_clast_direct_kernel[(N * OD * OH, tiles_ow)](...)
else:
    _softmax_pool2_clast_persistent_kernel[(_MAX_GRID,)](..., total_tiles, _MAX_GRID, tiles_ow, ...)
```

Ascend FFTS grid dimensions must stay within 65,535. The target shape uses the direct path, while very large shapes use a persistent tile loop over logical tiles rather than elements.

## 4. Safe ACL fallback for wide channel counts and large tile counts

```python
if (not _USE_TRITON_FUSED) or C > 64 or total_tiles > _ACL_TILE_THRESHOLD:
    return _acl_reference_post(x, int(pool_kernel_size))
```

The fused Triton path pads channels to a compile-time block. Wider channel counts or large production tile counts can exceed the practical custom-kernel/compile budget. The optimized host keeps correctness and target latency robust by routing those cases to native PyTorch/CANN operators while retaining the Triton direct and persistent kernels for small/medium tested paths.
