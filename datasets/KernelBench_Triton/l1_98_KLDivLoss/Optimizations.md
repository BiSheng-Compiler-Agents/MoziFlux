# Optimizations for 98_KLDivLoss

## 1. Size-based Triton dispatch

```python
if B * D >= 50_000_000:
    _kl_div_row_contig_kernel[(n_programs,)](...)
    return row_sums.sum() / B
_kl_div_row_atomic_kernel[(n_programs,)](...)
return out[0]
```

Small and medium shapes avoid the extra `row_sums.sum()` launch by atomically accumulating one scalar per row. The exact large shape keeps the row-sum path because hardware verification showed many-row atomics regress at that scale.

## 2. Contiguous flattened row addressing

```python
base = row * D
idx = base + cols
p = tl.load(pred_ptr + idx, mask=mask, other=1.0)
t = tl.load(targ_ptr + idx, mask=mask, other=0.0)
```

The source kernel passes runtime strides after making inputs contiguous. The optimized row kernels use flattened contiguous offsets, preserving coalesced GM loads and reducing address-generation state.

## 3. Grid-cap safe persistent row loop

```python
n_programs = min(B, 65535)
for row in range(pid, B, n_programs):
    ...
```

The optimized kernels preserve correctness for row counts above Ascend's 65,535 launch cap while keeping the normal target shape at one program per row.

## 4. Diagnostic two-phase fallback

```python
n_tiles = triton.cdiv(n_elements, _BLOCK)
n_programs = min(n_tiles, 65535)
_kl_div_partial_kernel[(n_programs,)](...)
_kl_div_finalize_kernel[(triton.cdiv(n_programs, _FINAL_BLOCK),)](...)
```

This non-default fallback covers the full-tensor partial-reduction dispatch path in unit tests and cannsim without replacing the faster production dispatch.
