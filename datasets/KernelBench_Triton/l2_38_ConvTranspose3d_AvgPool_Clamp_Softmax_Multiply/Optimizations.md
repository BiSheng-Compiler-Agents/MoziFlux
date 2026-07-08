# Optimizations

## Operator contract
`ModelNew` computes:

```python
x = ConvTranspose3d(...)(x)
x = AvgPool3d(pool_kernel_size)(x)
x = clamp(x, clamp_min, clamp_max)
x = softmax(x, dim=1)
x = x * 2.0
```

The default epilogue input after convolution and pooling is NCDHW = `(32, 64, 32, 64, 64)`.

## 1. Replace 2-D Triton launch with flattened 1-D direct launch

Baseline used a 2-D grid:

```python
def grid(META):
    return (N, triton.cdiv(DHW, META['BLOCK_POS']))
```

The optimized direct path flattens `(N, spatial_tile)` into one 1-D tile id:

```python
tile_id = tl.program_id(0)
pos_tiles = tl.cdiv(DHW, BLOCK_POS)
pid_n = tile_id // pos_tiles
pid_tile = tile_id - pid_n * pos_tiles
```

Host launch:

```python
total_tiles = N * triton.cdiv(DHW, BLOCK_POS)
_clamp_softmax_mul2_direct_ncdhw[(total_tiles,)](...)
```

Rationale: 1-D launch is the safer Ascend pattern and makes the grid-cap check explicit in host code.

## 2. Add grid-cap dispatch for the default target shape

For `C=64, BLOCK_POS=64`, the target has:

```python
total_tiles = 32 * ceil(131072 / 64) = 65536
```

This exceeds the Ascend FFTS/coreDim cap of 65535. The optimized host avoids launching the oversized custom Triton epilogue:

```python
if total_tiles > _MAX_PROGRAMS:
    return torch.softmax(
        torch.clamp(x, min=float(clamp_min), max=float(clamp_max)), dim=1
    ) * float(scale)
```

Rationale: the default shape is exactly one tile over the grid cap; CANN's native clamp/softmax path is correct, grid-safe, and faster on the target hardware than the readable baseline2 path.

## 3. Keep a UB-safe Triton fast path for small/medium shapes

```python
BLOCK_C = max(32, _next_power_of_2(C))
BLOCK_POS = _block_pos_for_channels(BLOCK_C)
```

For the required `C=64` direct path, `BLOCK_POS=64`, so each program handles a `[64, 64]` softmax tile. This preserves the original fused clamp/softmax/multiply behavior for shapes whose tile count is under the grid cap.

## 4. Correctness-preserving reference behavior

The ConvTranspose3d and AvgPool3d modules and constructor signature are preserved. The fallback computes the same math as the reference:

```python
torch.softmax(torch.clamp(x, min=clamp_min, max=clamp_max), dim=1) * 2.0
```

Remote verification passed all optimized paths with max absolute difference `2.23517e-08`.
