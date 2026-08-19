# BatchNorm HW-block reduction + persistent dispatch

## When this applies

BatchNorm2d over contiguous NCHW float32 where the baseline reduces one partial per `(C, N, H)` row. At large target shapes, `C * N * H` can exceed Ascend FFTS `coreDim <= 65535`, and the row-wise partial tensor becomes large.

## Pattern

1. Require/validate contiguous NCHW so each `(n, c)` plane is a flat contiguous `H*W` region.
2. Reduce larger contiguous `H*W` blocks, e.g. `BLOCK_SIZE=2048`, producing partials shaped `(C, N * ceil(H*W / BLOCK_SIZE))`.
3. Finalize per channel in fp32: `mean=sum/M`, `var=max(sumsq/M - mean*mean, 0)`, `invstd=rsqrt(var+eps)`.
4. Apply using the same HW-block tiling.
5. Use direct launch when `C * N * ceil(HW/BLOCK) <= 65535`; otherwise use a separate persistent JIT kernel capped at `65535` programs.

```python
NUM_HW_BLOCKS = triton.cdiv(H * W, BLOCK_SIZE)
NUM_PARTS = N * NUM_HW_BLOCKS
total_tiles = C * NUM_PARTS
if total_tiles > 65535:
    _reduce_persistent[(65535,)](..., 65535, total_tiles, NUM_PARTS, ...)
else:
    _reduce_direct[(C, NUM_PARTS)](...)
```

## Why it helps

Compared with row-wise `W` reduction, HW-block reduction reduces partial GM traffic and avoids invalid launch grids. In the observed BatchNorm case, cannsim per-element reduction efficiency improved from `3373 cycles / 512 elems = 6.59 cycles/elem` to `3487 cycles / 2048 elems = 1.70 cycles/elem` (~3.87x), while remote hardware correctness passed with max error <= `1.43e-6`.

## Profiling guard

Do not launch comparison baselines that exceed the Ascend grid cap during either unit tests or benchmarks; an invalid launch aborts the process and poisons the verifier. Keep the provider column and print a `SKIP`/`inf` guard, while still requiring optimized correctness.
