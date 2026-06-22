# Optimizations

## 1. Production dispatch to ACL cumsum

```python
y = torch.empty((B - 1, N + 1), device=x.device, dtype=x.dtype)
y[:, 0].zero_()
y[:, 1:] = torch.cumsum(x[:-1], dim=1)
```

The baseline implements a serial Triton prefix loop per row. `torch.cumsum` uses the mature Ascend library scan path and avoids the per-element Triton scalar loop, while preserving the padded exclusive output contract: leading zero plus inclusive cumulative sum of `x[:-1]`.

## 2. Vectorized Triton fallback for cannsim and dispatch coverage

```python
x = tl.load(row_x_ptr + col * stride_x_col, mask=mask, other=0.0).to(tl.float32)
prefix = tl.cumsum(x, axis=0) + carry
tl.store(row_y_ptr + (col + 1) * stride_y_col, prefix, mask=mask)
```

This replaces the baseline inner `tl.static_range(BLOCK_SIZE)` scalar load/add/store loop with a single vector tile load, `tl.cumsum`, and vector store. It keeps one scalar carry between chunks so it generalizes to large `N` without hardcoding the target shape.

## 3. Grid-cap row loop for fallback legality

```python
n_programs = min(rows, 65535)
for row in tl.range(pid0, rows, n_programs):
    ...
```

The target row count is below the Ascend FFTS grid cap, but the fallback is made legal for larger valid 2D inputs by capping the launch grid and looping rows inside the kernel.
