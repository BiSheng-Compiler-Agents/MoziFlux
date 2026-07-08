# Pooling grid-cap dispatch pattern

For MaxPool/AvgPool style kernels that tile over an outer row dimension and a contiguous output-width dimension, the natural 2D launch product can exceed Ascend's `coreDim <= 65535` limit even when each axis looks reasonable.

General fix:
```python
MAX_PROGRAMS = 65535
BLOCK_W = 64
n_w_tiles = triton.cdiv(outW, BLOCK_W)
total_tiles = N * C * outD * outH * n_w_tiles

if total_tiles <= MAX_PROGRAMS:
    _pool_tile_kernel[(total_tiles,)](..., total_tiles, BLOCK_W=BLOCK_W)
else:
    _pool_persistent_kernel[(MAX_PROGRAMS,)](..., total_tiles, MAX_PROGRAMS, BLOCK_W=BLOCK_W)
```

Inside the kernel, recover the original logical coordinates from the flattened tile id:
```python
tile_id = tl.program_id(0)  # or loop variable in persistent path
w_tile = tile_id % n_w_tiles
row = tile_id // n_w_tiles
ow = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
```

Persistent path:
```python
pid = tl.program_id(0)
for tile_id in range(pid, total_tiles, n_programs):
    ...  # process one logical row/W tile
```

Review/profiling rules:
- Guard the product of logical launch dimensions, not only individual axes.
- Keep a direct path for small shapes; persistent dispatch below the cap is usually a regression.
- Unit-test both direct and persistent paths. Use a small synthetic persistent shape when the real target is too large for the verifier window.
- For pooling without indices, vectorize across the contiguous output-width dimension when the compiler accepts the live vector state, and use `-inf` (max) or correct neutral/count handling (avg) for padding masks.
- If output-width vectorization plus `KH*KW` unrolling exceeds BiSheng VF stack or hits HFusion collapse assertions, fall back to a scalar one-output-per-program body and keep the grid-cap/persistent dispatch fix; legality beats unshippable vectorization.
- For no-padding AvgPool fast paths, remove boundary/count work and divide by constant `KH*KW`; route padded cases to a correct fallback or a separate masked/counting kernel rather than reusing the no-padding body.

This pattern is a generalization of 1D/2D/3D pooling kernels where output tiles are independent and can be flattened safely.
  ```python
  cols_per_launch = max(1, 65535 // max(1, n_rows))
  for start in range(0, n_col_blocks, cols_per_launch):
      cols = min(cols_per_launch, n_col_blocks - start)
      _pool_cols_kernel[(n_rows, cols)](..., COL_BLOCK_START=start)
  ```
  This is a legality workaround, not guaranteed optimization: multiple launches can still lose badly to PyTorch/ACL, so let remote hardware latency decide rather than trusting grid=1 cannsim.
- If editable or golden comparison providers contain Ascend-unsupported code (for example `cache_modifier=".cg"`), keep them visible in `profile_kernels.py` as `inf`/`SKIP` with the reason; do not fabricate a Triton baseline speedup.

This pattern is a generalization of 1D/2D/3D pooling kernels where output tiles are independent and can be flattened safely, with AvgPool-specific caveats for division semantics and launch-overhead tradeoffs.
