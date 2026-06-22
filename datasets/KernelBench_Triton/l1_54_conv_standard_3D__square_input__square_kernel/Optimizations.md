# Optimizations

## 1. Removed redundant Triton no-op launch

Baseline code launched `_noop_touch_kernel` over the full 5D input and then returned `self.conv3d(x)`:

```python
n_elements = x.numel()
grid = (triton.cdiv(n_elements, 1024),)
_noop_touch_kernel[grid](x, n_elements, BLOCK=1024)
return self.conv3d(x)
```

The kernel result is unused and does not alter `x`, so it only adds Triton compilation/launch overhead before the ACL Conv3d path. The optimized host path preserves all constructor parameters and validation but directly dispatches Conv3d:

```python
if x.dim() != 5:
    raise ValueError(...)
if x.device.type != "npu":
    raise RuntimeError(...)
return self.conv3d(x)
```

## Rationale

- Correctness is preserved because the removed Triton kernel has no stores, atomics, or side effects used by the convolution.
- The exact source input contract remains unchanged: `(batch, channels, depth, width, height)` with square 3D kernel defaults from `get_init_inputs()`.
- cannsim confirms the no-op kernel has only scalar/flow-control setup work and no convolution computation; removing it eliminates that whole extra dispatch path.
