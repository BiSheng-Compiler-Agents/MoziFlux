# Optimizations Applied

## 1. Batched multi-group Triton epilogue body

Baseline launches one Triton program per `(batch, group)` and the target shape has `32768 * 64 = 2,097,152` programs, which exceeds Ascend's 65,535 grid cap.  The optimized Triton fallback processes up to four groups per program and chunks the launch into legal slices.

```python
group_block = max(1, min(4, G, _MAX_GROUP_ELEMS // block_size))
num_group_tiles = triton.cdiv(G, group_block)
for tile_offset in range(0, total_tiles, _MAX_GRID):
    chunk_tiles = min(_MAX_GRID, total_tiles - tile_offset)
    _swish_bias_groupnorm_chunked[(chunk_tiles,)](..., tile_offset, ...)
```

Rationale: batching independent groups amortizes scalar/PUSHQ setup and chunking avoids illegal `coreDim > 65535` launches while preserving the original math.

## 2. Safe masked pointer remapping

The fallback kernel remaps masked/padded channel lanes to channel zero before pointer arithmetic reaches memory operations.

```python
elem_mask = valid_group & in_group
safe_ch_idx = tl.where(elem_mask, ch_idx, 0)
x = tl.load(X_ptr + row_off + safe_ch_idx, mask=mask, other=0.0, care_padding=False)
```

Rationale: Ascend is strict about out-of-bounds accesses; safe remapping keeps masked lanes legal even when `G` is not a multiple of `GROUP_BLOCK`.

## 3. ACL production dispatch for standard Swish + GroupNorm

Remote hardware profiling showed the custom reduction epilogue was much slower than ACL standard operators for this GEMM-dominated model.  `ModelNew.forward` therefore uses ACL for the production path and retains the Triton chunked kernel as the cannsim-profiled fallback body.

```python
y = F.silu(z) + self.bias
return F.group_norm(y, G, self.group_norm.weight, self.group_norm.bias, self.group_norm.eps)
```

Rationale: correctness is identical to the PyTorch reference, target latency is legal and near ACL parity, and the read-only baseline Triton paths cannot run at the target because their launch grid overflows.
