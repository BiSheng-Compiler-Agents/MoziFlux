# Reverse cumsum: custom Triton scan vs ACL dispatch

## When to use

Use this when optimizing reverse cumulative sum / scan operators on Ascend where the baseline is a custom Triton row-wise scan using `tl.cumsum` over reversed tiles.

## Observed pattern

A custom Triton reverse-cumsum path can remain scalar-limited even after normal cleanup:

```python
for b in tl.range(0, NUM_BLOCKS):
    block_idx = NUM_BLOCKS - 1 - b
    x_rev = tl.load(...).to(tl.float32)
    scan_rev = tl.cumsum(x_rev, axis=0)
    tl.store(..., scan_rev + carry, mask=mask)
    carry += tl.sum(x_rev, axis=0)
```

On Ascend, `tl.cumsum` may lower to serial scan-like scalar load/store work rather than an efficient vector prefix primitive. Cannsim symptoms: SCALARLDST/PUSHQ dominate, with `ST_XD_XN_IMM`, `LD_XD_XN_IMM`, and `VF` among the top cycle consumers.

## Recommended decision process

1. First try the normal custom-kernel cleanup only as an A/B candidate: remove Python prefetch branches, use `tl.range`, keep masks, use a larger safe tile if UB allows, and cap row grids.
2. Run a bounded cannsim diagnostic probe if full `BLOCK_N` does not stabilize. For scan kernels, a tiny probe such as `BLOCK_N=8`, `NUM_BLOCKS=2`, `N=16`, `grid=(1,)` can still reveal whether SCALARLDST/PUSHQ remain dominant.
3. If the custom path remains scalar-limited and the operation is semantically a standard scan, dispatch production traffic to ACL/PyTorch:

```python
def cumsum_reverse_npu(x, dim=1):
    dim = dim if dim >= 0 else x.ndim + dim
    if x.shape[dim] == 0:
        return torch.empty_like(x)
    return torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim), dims=[dim])
```

4. Keep provider columns in `profile_kernels.py`: PyTorch / ACL, editable baseline, read-only `base_*.py`, optimized. Preserve `base_*.py` as read-only and pre-skip/return `inf` for comparison-provider MLIR failures on non-target shapes.

## Reporting guidance

- State clearly that cannsim traces were for the custom fallback candidate, while production optimized hardware latency comes from ACL dispatch.
- Document failed full-tile cannsim attempts as scale limits, not as tool failure; use the bounded probe trace table for bottleneck evidence.
- Use hardware `remote_verify` latency for the final decision because dispatch/library effects are not visible in a single-core sub-kernel trace.
