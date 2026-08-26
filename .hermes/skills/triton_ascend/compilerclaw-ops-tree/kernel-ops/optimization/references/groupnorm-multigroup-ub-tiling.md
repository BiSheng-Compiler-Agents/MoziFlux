# GroupNorm multi-group UB tiling

When a fused GroupNorm epilogue already loads each group once and computes mean/variance in UB, the remaining bottleneck can be per-program dispatch/PUSHQ/scalar setup from launching one program per `(row, group)`. If `Cg` is moderate and multiple independent groups fit in UB, batch several groups for the same row in one 2D tile.

## Pattern

```python
GROUP_BLOCK = max(1, min(4, G, MAX_GROUP_ELEMS // BLOCK_SIZE))
groups_per_row = triton.cdiv(G, GROUP_BLOCK)

gidx = group_tile * GROUP_BLOCK + tl.arange(0, GROUP_BLOCK)
offs = tl.arange(0, BLOCK_SIZE)
offs_c = gidx[:, None] * Cg + offs[None, :]
mask = (row < N) & (gidx[:, None] < G) & (offs[None, :] < Cg)

x = tl.load(x_ptr + row * C + offs_c, mask=mask, other=0.0,
            care_padding=False).to(tl.float32)
mean = tl.sum(x, axis=1) / Cg
xc = x - mean[:, None]
var = tl.sum(xc * xc, axis=1) / Cg
out = xc * tl.rsqrt(var + eps)[:, None] * gamma + beta
```

Use a direct path while `N * groups_per_row <= 65535`; add a persistent tile loop only for overflow legality.

## UB and correctness notes

- Cap live elements, e.g. `GROUP_BLOCK * BLOCK_SIZE <= 2048`, unless a fresh UB estimate proves larger is safe.
- This is not a replacement for single-pass GroupNorm; it is the next optimization after single-pass is already present.
- Cover both direct and persistent paths in `profile_kernels.py`; use a correctness-only synthetic persistent shape if the real target does not overflow.
- Compare cannsim per normalized group when baseline and optimized programs process different group counts.

## Session data point

For a GEMM->GroupNorm->HardTanh epilogue with `G=16, Cg=512`, batching four groups changed cannsim from 6013 cycles for one group to 4429 cycles for four groups (1107 normalized cycles/group). Remote hardware improved editable baseline1 target latency 6.787034 ms -> 6.499175 ms and geomean 1.247x across small/medium/irregular/target shapes.
