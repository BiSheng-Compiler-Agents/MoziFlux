# Optimizations

## 1. Removed Ascend-unsupported cache modifier

Baseline:
```python
tl.load(..., cache_modifier=".cg")
```

Optimized:
```python
vals = tl.load(x_row + safe_pos, mask=valid, other=0.0)
```

Rationale: `.cg` is unsafe on Triton-Ascend and both editable/golden Triton baselines fail remote compilation with `unsupported_cache_modifier_cg`. Plain masked loads compile and preserve padding semantics.

## 2. Grid-safe column tiling for normal row counts

Optimized host chunks the output-column grid so each launch respects Ascend `coreDim <= 65535`:
```python
cols_per_launch = max(1, 65535 // max(1, n_rows))
while start < n_col_blocks:
    cols = min(cols_per_launch, n_col_blocks - start)
    _avgpool1d_cols_kernel[(n_rows, cols)](..., COL_BLOCK_START=start)
    start += cols
```

Rationale: direct `(N_ROWS, N_COL_BLOCKS)` is illegal for the target product (`8192 * 257`), while one-row kernels serialize all columns. Chunking keeps the 2-D tile mapping legal.

## 3. Row-capped fallback for very high row counts

When `B*C > 65535`, the optimized host falls back to a capped row loop:
```python
_avgpool1d_row_kernel[(65535,)](..., N_PROGRAMS=65535)
```

Rationale: this covers the high-row dispatch path without poisoning the NPU context with an illegal 2-D launch.

## 4. FP32 accumulation with exact AvgPool1d padding semantics

```python
valid = (pos >= 0) & (pos < L_IN) & mask_o
safe_pos = tl.minimum(tl.maximum(pos, 0), L_IN - 1)
acc += tl.load(x_row + safe_pos, mask=valid, other=0.0).to(tl.float32)
tl.store(y_row + offs, acc * (1.0 / float(KERNEL_SIZE)), mask=mask_o)
```

Rationale: invalid padded positions contribute zero, but the divisor remains `KERNEL_SIZE`, matching `nn.AvgPool1d(..., count_include_pad=True, ceil_mode=False)`.
