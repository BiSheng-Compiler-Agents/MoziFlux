# 3D max reduction tiling pattern

Use this for 3D tensors reduced over one dimension, especially target shapes like `[B, M, N] -> [B, N]` where the reduced dimension is large and the kept dimension is contiguous.

## Pattern

Prefer a 2D load tile plus vector reduction over serial row updates:

```python
m_offsets = m_start + tl.arange(0, BLOCK_M)
n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
ptrs = x_ptr + b * sx0 + m_offsets[:, None] * sx1 + n_offsets[None, :] * sx2
mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
vals = tl.load(ptrs, mask=mask, other=-float("inf"), cache_modifier=".cg").to(tl.float32)
acc = tl.maximum(acc, tl.max(vals, axis=0))
```

For the inverse case (`dim=2`, output `[B, M]`), load `[BLOCK_M, BLOCK_N]` and reduce with `tl.max(vals, axis=1)`.

## Dispatch

Flatten logical 2D grids into a 1D grid capped by physical vector cores and Ascend `coreDim`:

```python
total_tiles = B * triton.cdiv(N, BLOCK_N)
n_programs = min(total_tiles, num_vector_cores, 65535)
_kernel[(n_programs,)](..., total_tiles, n_programs)

# device
for tile_id in tl.range(pid, total_tiles, n_programs):
    ...
```

This avoids `coreDim > 65535` for alternate dims and can reduce launch scheduling overhead. Use `tl.range`, not Python `range`, for the grid-stride loop.

## When it wins

This is most useful when a baseline performs many scalar/static row updates (e.g. `for mi in tl.static_range(...): load one row; acc=max(acc,row)`). The 2D tile reduction cuts scalar/control pressure and maps the local reduction to vector instructions.

## Argmax / index-return variant

For `argmax`, keep the value reduction and index reduction separate so PyTorch first-index tie semantics are explicit:

```python
tile_max = tl.max(vals, axis=0)
eq = (vals == tile_max[None, :]) & mask
idxs = tl.where(eq, m_offsets[:, None].to(tl.int32), M)
tile_idx = tl.min(idxs, axis=0)
update = (tile_max > best_val) | ((tile_max == best_val) & (tile_idx < best_idx))
```

Use int32 indices inside the kernel and cast to int64 only at the final store. Index-return reductions can become RVEC/MTE3-store heavy; do not assume a 2D tile that looks better per output in cannsim will beat ACL on full hardware. Keep the direct/ACL reference in the profile, report target latency honestly, and treat grid-capped Triton as a legality/correctness improvement unless hardware proves a speedup.

## Verification notes

- Test all dim dispatch paths (`dim=0/1/2` and negative aliases), not just the default `get_init_inputs()` dim.
- Keep all loads/stores masked; use `-inf` for masked max-reduction lanes.
- Use fp32 accumulation for fp16/bf16/fp32 inputs, then cast to output dtype.
- For argmax/index outputs, include tie cases and verify first-index behavior.
- In cannsim, compare a sub-kernel such as one batch and one output block; normalize by outputs-per-program when baseline and optimized produce different numbers of output elements. Dispatch-level savings from grid flattening may require hardware verification.
