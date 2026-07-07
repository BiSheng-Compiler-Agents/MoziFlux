# GEMM + Swish + bias + GroupNorm ACL dispatch

Use this pattern for `Linear/GEMM -> Swish/SiLU -> add bias -> GroupNorm` operators on Ascend.

## What to check

1. Treat ACL/PyTorch standard ops as a first-class optimized provider.  For this chain, `F.silu(z) + bias` followed by `F.group_norm(...)` can beat a custom Triton reduction epilogue by orders of magnitude on real hardware.
2. If the baseline Triton epilogue launches one program per `(batch, group)`, guard it before timing: `batch * groups > 65535` exceeds the Ascend launch grid cap and can poison the NPU context.
3. If a Triton epilogue is still useful for diagnostics or small shapes, batch independent groups per program and chunk host launches instead of relying on a persistent work-stealing loop when correctness is not proven.

## Production host pattern

```python
z = self.matmul(x)
y = F.silu(z) + self.bias
return F.group_norm(
    y,
    self.group_norm.num_groups,
    self.group_norm.weight,
    self.group_norm.bias,
    self.group_norm.eps,
)
```

## Legal Triton fallback pattern

```python
group_block = max(1, min(4, G, MAX_GROUP_ELEMS // block_size))
num_group_tiles = triton.cdiv(G, group_block)
total_tiles = B * num_group_tiles

for tile_offset in range(0, total_tiles, MAX_GRID):
    chunk_tiles = min(MAX_GRID, total_tiles - tile_offset)
    _epilogue_chunked[(chunk_tiles,)](..., tile_offset, ...)
```

Inside the fallback kernel, remap masked/padded channel lanes before pointer arithmetic reaches memory operations:

```python
elem_mask = valid_group & in_group
safe_ch_idx = tl.where(elem_mask, ch_idx, 0)
x = tl.load(ptr + row_off + safe_ch_idx, mask=mask, other=0.0)
```

## Reporting guidance

In `performance_report.md`, separate cannsim trace data for the Triton fallback body from remote hardware latency for the production ACL dispatch.  Do not claim that tile-level cannsim speedup explains an ACL-dispatch win; state that cannsim validates/diagnoses the fallback body while hardware results select the dispatch path.
