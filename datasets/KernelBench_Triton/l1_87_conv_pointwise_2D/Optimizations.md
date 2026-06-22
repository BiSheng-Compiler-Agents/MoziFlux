# Optimizations

## 1. Removed custom Triton 1x1-convolution launch

Baseline host path materializes a transposed weight and launches `_pw_conv1x1_kernel` over a 2D grid:

```python
wt = self.conv1d.weight.view(C_out, C_in).t().contiguous().to(dtype=x.dtype)
_pw_conv1x1_kernel[grid](x, wt, bias if has_bias else None, y, ...)
```

Optimized path preserves the same `ModelNew` constructor/parameters but dispatches the mature Ascend ACL Conv2d implementation:

```python
c = self.conv1d
weight = c.weight if c.weight.dtype == x.dtype else c.weight.to(dtype=x.dtype)
return F.conv2d(x, weight, bias, stride=c.stride, padding=c.padding,
                dilation=c.dilation, groups=c.groups)
```

Rationale: pointwise Conv2d is a standard ACL-covered primitive. The baseline custom kernel performs the operation as a GEMM-like custom launch and also risks Ascend `coreDim` overflow at the exact shape (`ceil(16*1024*1024/512) * ceil(128/64) = 65536` for one autotune candidate), while ACL handles tiling/dispatch internally.

## 2. Preserved dtype compatibility without changing module state

```python
weight = c.weight if c.weight.dtype == x.dtype else c.weight.to(dtype=x.dtype)
bias = None if c.bias is None else (c.bias if c.bias.dtype == x.dtype else c.bias.to(dtype=x.dtype))
```

Rationale: the baseline accepted fp16/fp32/bf16 inputs and converted the temporary weight tile to the input dtype. The optimized implementation keeps the learned parameter ownership unchanged and only creates a per-forward typed view when required.

## 3. Eliminated full-shape custom grid overflow path

The optimized path has no Triton launch grid. This removes custom-kernel dispatch from the exact target shape and avoids autotune candidates whose grid product reaches or exceeds the Ascend 65,535 launch limit.
