# Conv2d + InstanceNorm/Divide: preserve direct path, add persistent overflow path

## When this applies

- Main operator is ACL/PyTorch Conv2d, followed by a custom Triton row-wise per-plane normalization/scale epilogue.
- The direct fused Triton epilogue is already hardware-proven faster than ACL InstanceNorm for medium/target shapes.
- Normal target row count (`N * C_out`) is below Ascend `coreDim <= 65535`, but larger valid batches/channels can exceed it.

## Pattern

Keep the direct path unchanged for normal shapes; add a separate persistent row-loop only for legality/generalization:

```python
_MAX_PROGRAMS = 65535

total_rows = N * C
if total_rows > _MAX_PROGRAMS or x.numel() > 2_000_000_000:
    n_programs = min(total_rows, _MAX_PROGRAMS)
    _norm_persistent[(n_programs,)](..., n_programs, BLOCK_HW=block_hw)
else:
    _norm_direct[(total_rows,)](..., BLOCK_HW=block_hw)
```

Persistent kernel loops over rows and uses int64 offsets only on the overflow-safe path:

```python
row = tl.program_id(0)
while row < total_rows:
    ptrs = x_ptr + (row * hw + offs).to(tl.int64)
    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    ...
    tl.store(ptrs, out.to(dtype), mask=mask)
    row += n_programs
```

For the direct path, keep int32-style addressing if hardware shows int64 offsets regress or do not help:

```python
ptrs = x_ptr + base + offs
```

## Profiling guidance

- Do not replace a measured-fast fused direct Triton normalization with ACL just because Conv2d itself is ACL-covered; benchmark first.
- Grid=1 cannsim for direct vs optimized may be near-identical; the persistent path primarily fixes launch legality.
- Add a correctness-only synthetic shape with `N * C_out > 65535` and small spatial extent to exercise the persistent path without making it a misleading benchmark shape.
- In `profile_kernels.py`, keep baseline provider columns visible and print neutral `SKIP grid_guard` for direct baselines on the synthetic overflow shape.

## Example evidence

For a Conv2d + InstanceNorm2d + divide case, target `N*C_out = 16,384` stayed below the grid cap and the direct fused Triton path was faster than ACL InstanceNorm at medium/exact sizes. A persistent-only rewrite or int64-address direct rewrite did not improve target latency, so the successful optimization was to preserve direct target behavior and add a grid-capped persistent fallback for larger valid row counts.
