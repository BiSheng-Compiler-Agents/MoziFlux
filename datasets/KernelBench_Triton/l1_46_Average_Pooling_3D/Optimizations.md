# Optimizations Applied

## 1. Flattened 3D pooling work into row/width tiles

**Before** (`46_Average_Pooling_3D.py`): each program handles a flat `BLOCK=256` vector of output elements and repeatedly unravels `n,c,od,oh,ow` per lane.

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
ow = offs % OW
t = offs // OW
oh = t % OH
```

**After** (`opt_46_Average_Pooling_3D.py`): each program owns one `(N,C,OD,OH)` row and a contiguous `OW` tile.

```python
w_tile = tile_id % n_w_tiles
row = tile_id // n_w_tiles
ow = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
```

Rationale: average-pooling windows are independent, and vectorizing over contiguous output width removes most per-element unraveling/division work while keeping GM loads contiguous.

## 2. Grid-cap persistent dispatch for Ascend `coreDim <= 65535`

The source shape has `16*32*64*64*128 = 268,435,456` output elements, so the baseline launch `cdiv(n_elements, 256)=1,048,576` exceeds the Ascend FFTS grid limit.

```python
total_tiles = N * C * OD * OH * triton.cdiv(OW, _BLOCK_W)
if total_tiles <= _MAX_PROGRAMS:
    _avgpool3d_tile_kernel[(total_tiles,)](...)
else:
    _avgpool3d_persistent_kernel[(_MAX_PROGRAMS,)](..., total_tiles, _MAX_PROGRAMS, ...)
```

Rationale: the direct path remains fastest for small shapes, while the persistent path makes the target-sized dispatch legal by looping over logical tiles inside a capped grid.

## 3. Constant divide and masked fp32 accumulation preserved

```python
acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
vals = tl.load(x_ptr + row_base + iwv, mask=m, other=0.0)
acc += vals.to(tl.float32)
scale = 1.0 / float(KSIZE_D * KSIZE_H * KSIZE_W)
tl.store(y_ptr + out_idx, acc * scale, mask=mask_ow)
```

Rationale: `AvgPool3d` defaults to `count_include_pad=True`; invalid padding positions must contribute zero while the divisor remains `KD*KH*KW`. FP32 accumulation preserves precision for fp16/fp32 inputs.
