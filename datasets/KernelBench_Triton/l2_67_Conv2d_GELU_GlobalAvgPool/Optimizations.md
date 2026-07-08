# Optimizations Applied

## 1. Removed the custom Triton GELU + global-average-pool epilogue

**Before** (`67_Conv2d_GELU_GlobalAvgPool.py`):

```python
x = self.conv(x)
return gelu_global_avg_pool2d_triton(x)
```

The baseline convolution already uses ACL/PyTorch, then launches a vector-core Triton kernel that serially loops over each `(N, C)` spatial plane, computes exact GELU with `tl.erf`, reduces `H*W`, and stores one scalar.

**After** (`opt_67_Conv2d_GELU_GlobalAvgPool.py`):

```python
y = F.conv2d(x, weight, bias=bias, stride=stride,
             padding=padding, dilation=dilation, groups=groups)
return F.gelu(y, approximate="none").mean(dim=(-2, -1))
```

**Rationale:** GELU and spatial mean are standard ACL-covered operations.  Routing the epilogue through ACL removes an extra custom Triton launch, avoids per-plane scalar loop overhead, and preserves the exact PyTorch GELU formula used by the baseline.

## 2. Preserved constructor and tensor contracts

```python
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
```

The optimized `ModelNew` keeps the same initialization inputs, convolution parameters, `get_inputs()`, and `get_init_inputs()` as the editable input file.  No new shape restrictions were added.

## 3. Profiling safeguards for read-only / sandboxed providers

`profile_kernels.py` keeps parser-visible columns for PyTorch/ACL, Baseline Triton1, Baseline Triton2, and Optimized Triton.  Because the sandbox forbids reading `base_*.py`, Baseline Triton2 is reported as `SKIP_COMPARISON`/`inf` without importing it.
