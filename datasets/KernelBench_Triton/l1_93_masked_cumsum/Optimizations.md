# Optimizations

## 1. Remove custom Triton row-scan from production path

Baseline code materializes a masked tensor and then launches a custom Triton row-wise scan:

```python
y = x * mask.to(dtype=x.dtype)
_cumsum_lastdim_kernel[(m_size,)](...)
```

The optimized path dispatches the standard Ascend ACL/PyTorch scan primitive directly:

```python
return torch.cumsum(x * mask.to(dtype=x.dtype), dim=dim)
```

Rationale: masked cumsum is a mature library-covered scan operation. Cannsim shows the custom `tl.cumsum` body remains scalar/SCALARLDST limited, so preserving a custom Triton launch adds slow scan code and launch/compile overhead without using a better hardware primitive.

## 2. Preserve baseline validation and general shape contract

```python
if x.shape != mask.shape:
    raise ValueError("x and mask must have the same shape")
if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
    raise TypeError(...)
dim = dim % x.ndim
```

Rationale: the optimized `ModelNew(dim=1)` remains valid for all original supported ranks, dtypes, and dimensions rather than only the benchmark shape `(32768, 32768)`.

## 3. Avoid dimension transposes and manual row flattening

Baseline moves non-last dimensions to the tail, forces contiguity, flattens to 2D, and later moves dimensions back. The optimized ACL call accepts `dim` directly:

```python
torch.cumsum(masked_x, dim=dim)
```

Rationale: removing host-side `movedim(...).contiguous()` and 2D reshaping avoids extra tensor movement for non-last-dimension use while delegating the scan schedule to ACL.
