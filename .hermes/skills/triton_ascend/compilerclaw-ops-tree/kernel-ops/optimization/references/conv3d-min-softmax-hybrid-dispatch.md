# Conv3d + min(D) + softmax(C) hybrid dispatch

Use this when the main `Conv3d` already runs through ACL/PyTorch and a following Triton epilogue computes `min` over depth then `softmax` over channels.

## Pattern

Keep the custom fused Triton post-op for small/medium post-conv tensors when hardware proves it wins, but benchmark ACL `torch.amin(..., dim=2)` + `F.softmax(..., dim=1)` as a first-class target-shape provider. A cannsim per-tile improvement can still regress full-shape hardware if it increases logical program count or if ACL reductions are more efficient at scale.

```python
x = self.conv(x).contiguous()
_, C, _, H, W = x.shape
total_tiles = x.shape[0] * H * triton.cdiv(W, BLOCK_W)
if _next_pow2(C) <= 64 and total_tiles < ACL_TILE_THRESHOLD:
    return _fused_min_softmax_triton(x)
return F.softmax(torch.amin(x, dim=2), dim=1)
```

## Dispatch and tests

- Include a direct fused Triton path for legal small/medium shapes.
- Include a grid-capped persistent fused fallback if the fused path may see `total_tiles > 65535`.
- Include an ACL fallback for wider channel counts that do not fit the fused C tile.
- Add a unit-test-only wide-channel shape (for example `C_out > 64`) to prove the ACL fallback, without necessarily putting it in the benchmark table.

## Reporting

- Report cannsim traces for the retained fused Triton path, but separate that from target production dispatch when the target routes to ACL.
- If tile-size tuning improves cannsim but worsens hardware due extra programs, keep the hardware-winning tile/dispatch and document the discrepancy.
- Benchmark columns should keep all providers visible; correctness must pass for optimized small/medium/target plus fallback coverage.
