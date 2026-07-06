# Optimizations Applied

## Kernel
- Input: `48_Conv3d_Scaling_Tanh_Multiply_Sigmoid.py`
- Output: `opt_48_Conv3d_Scaling_Tanh_Multiply_Sigmoid.py`
- Operation preserved: `Conv3d -> multiply by per-channel scaling_factor -> tanh -> multiply by per-channel bias -> sigmoid`.

## 1. Channel-segment tiling replaces per-element channel indexing

Baseline computes the channel index for every element:

```python
c_idx = (offs // DHW) % C
sf = tl.load(sf_ptr + c_idx, mask=m, other=0.0)
b = tl.load(bias_ptr + c_idx, mask=m, other=0.0)
```

The optimized direct path assigns each program to one `(N, C, DHW-tile)` segment, so the channel is scalar per program:

```python
seg = pid // tiles_per_segment
tile_in_seg = pid - seg * tiles_per_segment
hw = tile_in_seg * BLOCK_SIZE + offs
c = seg % 16
base = seg * DHW + hw
sf = tl.load(sf_ptr + c)
b = tl.load(bias_ptr + c)
```

Rationale: cannsim showed the baseline was scalar/scalar-LDST dominated by per-lane `DIV`, `REM`, `ST_XD_XN_IMM`, and masked per-element parameter loads. Scalarizing `c` removes per-element division/remainder and turns scaling/bias into one scalar load each.

## 2. Direct + persistent dispatch under the Ascend grid cap

```python
total_tiles = total_segments * tiles_per_segment
grid = (min(total_tiles, _MAX_GRID),)
use_persistent = total_tiles > _MAX_GRID
```

For normal shapes (`total_tiles <= 65535`) the direct kernels launch one program per tile. For larger legal inputs, persistent kernels grid-stride over tiles:

```python
for tile_id in tl.range(pid, total_tiles, nprog, num_stages=2):
    ...
```

Rationale: the default benchmark uses the faster direct path, while large tensors avoid Ascend FFTS `coreDim > 65535` failures. `profile_kernels.py` unit-tests all four optimized dispatch paths: C=16 direct, C=16 persistent, generic-C direct, and generic-C persistent.

## 3. Contiguous tile offsets and padding-care removal

```python
offs = tl.arange(0, BLOCK_SIZE)
tl.multiple_of(offs, 16)
tl.max_contiguous(offs, BLOCK_SIZE)
x = tl.load(x_ptr + base, mask=mask, other=0.0, care_padding=False)
```

Rationale: Conv3d output is made contiguous before the epilogue. The optimized kernel uses contiguous HW tile offsets, enabling simpler MTE transactions and avoiding extra padding checks for masked tail lanes.

## 4. Preserved generic fallback

The optimized file contains generic-C direct and persistent kernels using `c = seg % C` when `C != 16`, without adding new runtime restrictions.

```python
elif not use_persistent:
    _fused_pointwise_ncdhw_direct_generic_kernel[grid](... C ...)
else:
    _fused_pointwise_ncdhw_persistent_generic_kernel[grid](... C ...)
```

Rationale: the problem default is `out_channels=16`, but the constructor still accepts other channel counts; the optimized host keeps them supported.
