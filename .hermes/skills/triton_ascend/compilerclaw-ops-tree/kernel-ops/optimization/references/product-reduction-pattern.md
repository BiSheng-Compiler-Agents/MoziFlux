# Product reduction pattern on Triton-Ascend

## Problem

`tl.prod` may be unavailable in triton-ascend. Row-streaming product reductions over a dimension also create many MTE/control events when the reduction dimension is large.

## Preferred pattern

For `[B, M, K] -> [B, K]` product over `M`, load a `[BLOCK_M, BLOCK_K]` tile, compute a prefix product along `M` with `tl.cumprod`, select the last valid row, and multiply the per-block result into a 1D accumulator.

```python
@triton.jit
def _prod_dim1_block_kernel(..., BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_k_base = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), 16), BLOCK_K)
    acc = tl.full([BLOCK_K], 1.0, dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m_idxs = m0 + offs_m
        vals = tl.load(
            x_ptr + b * stride_b + m_idxs[:, None] * stride_m + offs_k[None, :] * stride_k,
            mask=(m_idxs[:, None] < M) & mask_k[None, :],
            other=1.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        scan = tl.cumprod(vals, axis=0)
        last_rel = tl.minimum(BLOCK_M, M - m0) - 1
        block_prod = tl.sum(tl.where(offs_m[:, None] == last_rel, scan, 0.0), axis=0)
        acc *= block_prod
```

Why this is correct:
- `other=1.0` is the neutral value for masked `M`/`K` tails.
- Selecting the last prefix-product row preserves signs and avoids invalid reductions through sums/logs.
- fp16/bf16/fp32 inputs should be converted to fp32 for the reduction accumulator, then stored to output dtype.

## Dispatch and profiling notes

Use a 1D grid over output tiles and cap `n_programs = min(total_tiles, 65535)`, then loop `tile += n_programs` inside the kernel.

If the original baseline uses dynamic `while` row-streaming and the compiler aborts during cannsim, do not fabricate baseline trace data. Use either (a) a compileable representative of the same algorithm and label it explicitly as representative, or (b) report the compiler blocker and rely on hardware/provider columns for final comparison.
