# ConvTranspose + Channel Bias + Tanh Epilogue Pattern

Use this reference for `ConvTranspose2d` followed by per-channel bias/subtract/add and `tanh`/similar activation on NCHW output.

## Recognition pattern

- The transposed convolution is already handled by ACL/PyTorch (`F.conv_transpose2d` or `nn.ConvTranspose2d`).
- The custom Triton part is a contiguous NCHW epilogue such as `tanh(conv_out - bias[C,1,1])`.
- A flat elementwise kernel computes the channel for every element with `(offs // HW) % C`.
- The default output may be huge enough that a direct flat grid exceeds Ascend's 65,535 FFTS grid cap or byte offsets cross 2 GiB.

## Optimization

Keep ConvTranspose on ACL and retile only the epilogue by `(N,C)` plane:

```python
_BLOCK_HW = 4096
_MAX_PROGRAMS = 65535

@triton.jit
def _epilogue_direct(x_ptr, b_ptr, y_ptr,
                     HW: tl.constexpr, C: tl.constexpr,
                     TOTAL_TILES: tl.constexpr, NUM_HW_TILES: tl.constexpr,
                     BLOCK_HW: tl.constexpr):
    tile_id = tl.program_id(0)
    hw_tile = tile_id % NUM_HW_TILES
    plane = tile_id // NUM_HW_TILES
    c_idx = plane % C

    offs_hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = (tile_id < TOTAL_TILES) & (offs_hw < HW)
    offsets = plane.to(tl.int64) * HW + offs_hw.to(tl.int64)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    b = tl.load(b_ptr + c_idx, mask=tile_id < TOTAL_TILES, other=0.0)
    y = tl_tanh(x.to(tl.float32) - b.to(tl.float32)).to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)
```

Dispatch direct when `total_tiles <= _MAX_PROGRAMS`; otherwise launch a separate persistent kernel capped at `_MAX_PROGRAMS` that loops over tile IDs, not elements.

## Why it helps

- Removes per-element `// HW` and `% C` scalar work; channel is computed once per spatial tile.
- Preserves contiguous GM access over the spatial dimension.
- Keeps standard convolution work on ACL rather than attempting scalar/vector direct convolution.
- `int64` plane-base arithmetic protects late tiles when fp32 output byte offsets exceed 2 GiB.

## Cannsim notes

If the original baseline uses unsupported `tl.tanh`, record that the as-written baseline fails compilation. For bottleneck classification, a diagnostic micro-probe may replace only the API typo with `triton.language.math.tanh` while preserving the flat `(offs // HW) % C` indexing; label it as a diagnostic probe, not the shipped baseline. Compare normalized cycles/element when diagnostic and optimized probes use different `BLOCK_HW` sizes.

## Profiling notes

- Include direct-path, irregular non-power-of-two, default/huge, and forced-persistent unit tests.
- If comparison providers are invalid or read-protected, print `SKIP_UNAVAILABLE`/`INFO` rather than `FAIL` so optimized correctness remains the gate.
- Keep `Baseline Triton2` visible when `base_*.py` exists, but do not read/modify it in sandboxes that mark reference files as off-limits.
