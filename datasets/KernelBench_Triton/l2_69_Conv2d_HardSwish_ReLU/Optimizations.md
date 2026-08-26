# Optimizations

## 1. Algebraic ReLU(HardSwish) epilogue rewrite

Baseline materialized a ReLU intermediate and then clipped the scale term:

```python
rx = tl.maximum(x, 0.0)
y = rx * tl.minimum(rx * (1.0 / 6.0) + 0.5, 1.0)
```

The optimized kernel uses the exact piecewise identity for the final `ReLU(HardSwish(x))` result:

```python
y_pos = x * tl.minimum(x + 3.0, 6.0) * (1.0 / 6.0)
y = tl.where(x > 0.0, y_pos, 0.0)
```

Rationale: for `x <= 0`, the final ReLU makes the result zero; for `x > 0`, `HardSwish(x) = x * min(x + 3, 6) / 6`. This removes the explicit `tl.maximum(x, 0)` vector max chain and reduces RVECEX/PUSHQ work.

## 2. Legal direct + persistent dispatch

The optimized host keeps the fast direct launch for normal tensors and adds a grid-capped persistent fallback for tensors whose tile count exceeds Ascend FFTS' 65,535 grid limit:

```python
n_tiles = triton.cdiv(n_elements, block_size)
if n_tiles > _MAX_GRID:
    _hswish_relu_persistent_kernel[(_MAX_GRID,)](..., n_programs=_MAX_GRID)
else:
    _hswish_relu_direct_kernel[(n_tiles,)](...)
```

The persistent kernel iterates over tile IDs, not element IDs:

```python
n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
for tile_id in tl.range(pid, n_tiles, n_programs, num_stages=2):
    offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

Rationale: the default output has ~15.9k tiles at `BLOCK_SIZE=8192`, so direct dispatch remains fastest; larger accepted shapes remain legal instead of exceeding `coreDim <= 65535`.

## 3. Ascend-safe launch metadata and memory hints

The optimized launch uses `num_stages=2` and masked contiguous loads/stores with `care_padding=False`:

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
tl.store(y_ptr + offsets, y, mask=mask)
```

Rationale: `num_stages=1` is an Ascend pitfall, and the contiguous flat output from `nn.Conv2d` allows merged memory movement. Padding lanes do not participate in the stored output, so disabling padding care is safe for this pure elementwise epilogue.

## Results

- cannsim sub-kernel wall cycles: `4043 -> 3757` (`1.076x`, `-7.07%`).
- Remote hardware default-shape latency: `19.460703 ms -> 18.281796 ms` (`1.064x`, `-6.06%`).
- Unit tests: optimized direct and forced persistent paths passed with `max_abs=0`.
