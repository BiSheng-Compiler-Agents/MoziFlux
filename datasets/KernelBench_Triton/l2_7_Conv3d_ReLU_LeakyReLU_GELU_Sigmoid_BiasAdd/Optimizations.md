# Optimizations

## Operator
`Conv3d -> ReLU -> LeakyReLU -> GELU -> Sigmoid -> per-channel BiasAdd`.
The convolution remains on `nn.Conv3d`/ACL; only the contiguous NCDHW post-conv epilogue is custom Triton.

## 1. Channel-segment tiling instead of flat per-element channel recovery

**Baseline pattern**
```python
offs = block_start + tl.arange(0, BLOCK_SIZE)
c_idx = ((offs // stride_c) % C).to(tl.int32)
b = tl.load(bias_ptr + c_idx * bias_stride_c, mask=mask, other=0.0)
```
This computes channel for every vector lane, creating scalar `DIV`, `REM`, `SIGNEXT`, and scalar spill traffic.

**Optimized pattern**
```python
seg = tile_id // tiles_per_segment          # one scalar per program/tile
hw = tile_in_seg * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
base = seg * DHW + hw                       # contiguous within one N*C segment
c = (seg - (seg // C) * C).to(tl.int32)
b = tl.load(bias_ptr + c * bias_stride_c).to(tl.float32)
tl.store(y_ptr + base, x + b, mask=mask)
```
Rationale: NCDHW conv output is contiguous by spatial plane for each `(N,C)` segment. Tiling over `(N*C, DHW_tile)` makes the channel scalar for the whole tile and preserves contiguous GM load/store over `DHW`.

## 2. Direct + persistent grid-cap dispatch

```python
tiles_per_segment = triton.cdiv(dhw, _BLOCK_SIZE)
total_tiles = (n * c) * tiles_per_segment
grid_n = min(total_tiles, _MAX_GRID)
if total_tiles <= _MAX_GRID:
    _post_ops_channel_direct[(total_tiles,)](...)
else:
    _post_ops_channel_persistent[(grid_n,)](...)
```
The default shape has `DHW=115320`, `N*C=2048`, `BLOCK_SIZE=2048`, so `total_tiles=116736 > 65535`. The persistent path keeps the Ascend FFTS grid legal and loops over tiles rather than elements.

## 3. UB-safe block size

```python
_BLOCK_SIZE = 2048
```
The epilogue materializes several fp32 temporaries for GELU and sigmoid. `2048` elements keeps the working set within a conservative UB budget while giving contiguous DMA-sized vector work.

## 4. Preserve algebraic no-op and stable activation math

```python
x = tl.maximum(x, 0.0)
# LeakyReLU after ReLU is an exact no-op.
x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
x = 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))
```
Since ReLU makes `x >= 0`, LeakyReLU is exactly the identity. `exp2(-x*log2(e))` computes the same sigmoid branch for non-negative `x` and avoids the slower generic exponential form.
