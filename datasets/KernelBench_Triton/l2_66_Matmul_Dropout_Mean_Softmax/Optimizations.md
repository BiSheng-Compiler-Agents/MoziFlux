# Optimizations: 66_Matmul_Dropout_Mean_Softmax

## 1. Preserve the analytic constant-output shortcut

The operator is `Linear -> Dropout -> mean(dim=1, keepdim=True) -> softmax(dim=1)`.  Since the final tensor has shape `(batch, 1)`, softmax over that dimension returns `1` for every finite row, so the optimized host only allocates `(batch, 1)` and fills it.

```python
out = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)
_fill_ones_direct[(n_tiles,)](out, n_elements, BLOCK_SIZE=block, num_stages=2)
```

Rationale: avoids unused matmul/dropout/mean work and writes only the required output.

## 2. Remove unused module state from `ModelNew.__init__`

The optimized model keeps the same constructor signature but does not allocate `nn.Linear` and `nn.Dropout`, because their values cannot affect the finite output.

```python
self.in_features = in_features
self.out_features = out_features
self.dropout_p = dropout_p
```

Rationale: lower initialization memory/parameter overhead and no risk of accidental CPU/NPU parameter movement in the hot path.

## 3. Shape-aware direct fill tiling

Small/default outputs use `BLOCK_SIZE=128`; larger outputs use `BLOCK_SIZE=1024` to reduce launched programs and improve normalized store throughput.

```python
block = _FILL_BLOCK_SMALL if n_elements <= _FILL_BLOCK_SMALL else _FILL_BLOCK_LARGE
n_tiles = triton.cdiv(n_elements, block)
_fill_ones_direct[(n_tiles,)](out, n_elements, BLOCK_SIZE=block, num_stages=2)
```

Rationale: the default benchmark keeps one compact tile, while larger batches reduce launch-grid pressure by 8x versus a 128-element tile.

## 4. Grid-cap persistent fallback

A separate persistent kernel handles cases where `cdiv(n_elements, BLOCK_SIZE) > 65535`.

```python
if n_tiles > _MAX_PROGRAMS:
    _fill_ones_persistent[(_MAX_PROGRAMS,)](
        out, n_elements, _MAX_PROGRAMS, BLOCK_SIZE=block, num_stages=2
    )
```

Rationale: prevents Ascend `coreDim > 65535` failures while keeping the direct path for normal sizes.  `profile_kernels.py` force-tests this path by lowering `_MAX_PROGRAMS`.

## 5. Ascend-safe kernel launch metadata

The optimized launch uses `num_stages=2` and masks every store.

```python
mask = offs < n_elements
tl.store(out_ptr + offs, ones, mask=mask)
```

Rationale: avoids the known `num_stages=1` Ascend hazard and keeps out-of-bounds stores impossible.
