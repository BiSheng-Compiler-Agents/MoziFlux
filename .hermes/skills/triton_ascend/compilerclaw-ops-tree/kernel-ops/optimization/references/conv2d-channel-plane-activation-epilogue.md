# Conv2d channel-plane activation epilogue pattern

## When to use

Use for post-Conv2d epilogues on contiguous NCHW outputs where the epilogue applies per-channel parameters followed by pointwise activations, e.g.:

```python
y = activation2(activation1(conv(x) * per_channel_scale + per_channel_bias))
```

This is especially useful when a flat elementwise Triton kernel computes channel indices with per-element `offset // HW` and `% C`.

## Anti-pattern

Flat elementwise indexing over `N*C*H*W` makes channel lookup scalar-heavy:

```python
offsets = pid * BLOCK + tl.arange(0, BLOCK)
plane_idx = offsets // HW
c_idx = plane_idx % C
scale = tl.load(scale_ptr + c_idx, mask=mask, other=1.0)
```

Cannsim symptoms:
- `SCALAR` / `SCALARLDST` dominate.
- Top instructions include `DIV`, `REM`, `LD_XD_XN_IMM`, `ST_XD_XN_IMM`.
- Many trace events for a small diagnostic tile.

## Pattern

Map one program to each `(N, C)` output plane and loop over contiguous `HW` tiles. The channel is invariant inside the program, so scale/bias are loaded once.

```python
@triton.jit
def _channel_plane_epilogue(x_ptr, scale_ptr, y_ptr,
                            NC, HW, C,
                            BLOCK_HW: tl.constexpr):
    pid_nc = tl.program_id(0)
    c_idx = pid_nc % C
    scale = tl.load(scale_ptr + c_idx).to(tl.float32)
    offs = tl.arange(0, BLOCK_HW)
    base = pid_nc * HW
    for hw0 in tl.range(0, HW, BLOCK_HW, num_stages=2):
        hw = hw0 + offs
        mask = hw < HW
        x = tl.load(x_ptr + base + hw, mask=mask, other=0.0,
                    care_padding=False).to(tl.float32)
        v = x * scale
        # apply pointwise activation chain here
        tl.store(y_ptr + base + hw, v, mask=mask)
```

Host dispatch:

```python
_MAX_GRID = 65535
if NC <= _MAX_GRID:
    _channel_plane_epilogue[(NC,)](...)
else:
    _channel_plane_epilogue_persistent[(_MAX_GRID,)](..., n_programs=_MAX_GRID)
```

Persistent fallback loops over plane IDs, not elements:

```python
for pid_nc in tl.range(pid, NC, n_programs, num_stages=2):
    ...
```

## Verification requirements

- Unit-test both direct and persistent paths. For persistent, force it with a lowered `_MAX_GRID` instead of allocating huge tensors.
- Keep exact activation formulas unless tolerance explicitly permits approximations.
- Compare against the PyTorch/ACL chain; if ACL standard ops are faster, report that separately rather than hiding the optimized Triton improvement over the editable baseline.

## Observed evidence

In a Conv2d + per-channel multiply + LeakyReLU + exact GELU epilogue, this retiling changed a cannsim diagnostic tile from `7006` to `3034` wall cycles (`2.31x`) and reduced trace events from `1668` to `308`. Remote hardware target improved editable Baseline Triton1 from `1165.9 ms` to `168.7 ms` (`6.91x`), though the read-only ACL-style comparison remained faster.
