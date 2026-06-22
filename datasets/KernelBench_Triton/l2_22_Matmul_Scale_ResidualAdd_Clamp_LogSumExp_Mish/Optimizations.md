# Optimizations

## 1. Hybrid ACL/Triton post-op dispatch

```python
y = F.linear(x, weight, bias)
if y.device.type == "npu" and y.shape[1] <= 4096 and y.shape[0] <= 65535:
    return _launch_post_ops_triton(y, scale_factor, clamp_min, clamp_max)
return _torch_post_ops(y, scale_factor, clamp_min, clamp_max)
```

The target shape is a large GEMM followed by a hidden-dimension reduction. `F.linear` stays on ACL's tuned Cube implementation, and the large row reduction routes to ACL `torch.logsumexp`/elementwise ops instead of a scalar-heavy custom row kernel. The Triton post-op remains for smaller rows where launch overhead is acceptable and for cannsim-visible diagnostics.

## 2. Cleaner Triton fallback row kernel

```python
for bid in tl.range(0, n_blocks, 1):
    n_idx = bid * BLOCK_N + off_n
    vals = tl.load(base_ptr + n_idx * stride_y_n, mask=mask, other=0.0).to(tl.float32)
    ...
tl.store(out_ptr + pid_m * stride_out_m, out_val, mask=pid_m < B)
```

The fallback removes the unsupported `cache_modifier=".cg"`, replaces the Python-style `while` loop with `tl.range`, and masks the final store. Cannsim on a 1x1024 post-op tile improves from 10,460 to 9,807 wall cycles (6.2%).

## 3. Grid-cap fallback

```python
if y.shape[0] <= 65535:
    return _launch_post_ops_triton(...)
return _torch_post_ops(...)
```

The direct Triton row grid is only used while `gridX=B` is legal on Ascend. Larger batches fall back to ACL rather than risking `coreDim > 65535` or poisoned NPU context.
