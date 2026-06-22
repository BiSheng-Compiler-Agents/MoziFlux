# Optimizations

## 1. Preserve the fast direct fused InstanceNorm path

The target shape has `N*out_channels = 16,384`, safely under Ascend's 65,535 launch cap. The optimized direct path therefore preserves the baseline single-pass fused kernel:

```python
sum_x = tl.sum(vals_f32, axis=0)
sum_x2 = tl.sum(vals_f32 * vals_f32, axis=0)
scale = tl.rsqrt(var + eps) * (1.0 / div_const)
tl.store(ptrs, out.to(vals.dtype), mask=mask)
```

Rationale: hardware profiling showed the existing direct fused normalization/divide path is faster than ACL InstanceNorm for medium/exact shapes, so the production target path should not be replaced by a slower library sequence.

## 2. Add grid-capped persistent dispatch for large valid row counts

Baseline always launches one program per `(N,C)` plane:

```python
grid = (N * C,)
```

Optimized code keeps the direct path below the cap and routes oversized row counts to a separate persistent kernel:

```python
if total_rows > _MAX_PROGRAMS or x.numel() > 2_000_000_000:
    n_programs = min(total_rows, _MAX_PROGRAMS)
    _instancenorm_divide_persistent_kernel[(n_programs,)](..., n_programs, ...)
else:
    _instancenorm_divide_direct_kernel[(total_rows,)](...)
```

Inside the persistent kernel, each program loops over row tiles:

```python
row = tl.program_id(0)
while row < total_rows:
    ptrs = x_ptr + (row * hw + offs).to(tl.int64)
    ...
    row += n_programs
```

Rationale: this preserves target latency while making larger valid batches legal on Ascend; the unit test includes `persistent_rows` where baselines are skipped by `grid_guard` and optimized passes.

## 3. Use int64 offsets only on the overflow-safe path

The direct target path keeps baseline int32-style addressing, while the persistent path uses int64 offsets:

```python
# direct
ptrs = x_ptr + base + offs

# persistent
ptrs = x_ptr + (row * hw + offs).to(tl.int64)
```

Rationale: int64 offsets protect huge tensors from pointer-offset overflow, but keeping them out of the direct path avoids unnecessary target-shape overhead.
