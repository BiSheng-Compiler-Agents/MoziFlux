# Optimizations

## 1. Replace flat elementwise indexing with row/HW tiling

Baseline code computes the channel for every element:

```python
c_idx = (offs // HW) % C
b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
s = tl.load(scale_ptr + c_idx, mask=mask, other=0.0)
```

The optimized kernel maps each program to one `(N*C, HW-tile)` row tile, so the channel index is scalar for the whole tile:

```python
row = tile_id // N_HW_TILES
hw_tile = tile_id - row * N_HW_TILES
c_idx = row % C
b = tl.load(bias_ptr + c_idx).to(tl.float32)
s = tl.load(scale_ptr + c_idx).to(tl.float32)
```

Rationale: the baseline's per-element integer divide/mod generates scalar-load/store pressure; row tiling removes that scalar work and makes the main data load/store contiguous.

## 2. Add direct + persistent dispatch for the fused epilogue

```python
n_hw_tiles = triton.cdiv(HW, _BLOCK_HW)
total_tiles = N * C * n_hw_tiles
if total_tiles > _MAX_PROGRAMS:
    _bias_scale_sigmoid_row_persistent[(_MAX_PROGRAMS,)](..., total_tiles, _MAX_PROGRAMS, BLOCK_HW=_BLOCK_HW)
else:
    _bias_scale_sigmoid_row_direct[(total_tiles,)](..., total_tiles, BLOCK_HW=_BLOCK_HW)
```

Rationale: the exact shape produces `N*C*ceil(HW/4096)=65,536` row-tiles, just above Ascend's 65,535 launch cap. The direct path stays fastest for small/medium legal shapes; the persistent path fixes target/general oversized dispatch without changing math.

## 3. Remove autotune configs with risky grid and `num_stages=1`

The baseline autotune includes small `BLOCK_SIZE` configs that can exceed `coreDim` during tuning and an `8192` config with `num_stages=1`. The optimized kernel uses a fixed UB-safe `BLOCK_HW=4096` and explicit host routing.

Rationale: avoiding autotune-time invalid launches makes profiling deterministic and prevents NPU context poisoning at the target shape.

## 4. Preserve ACL convolution and GroupNorm

```python
x = self.conv(x)
x = fused_bias_scale_sigmoid(x, self.bias, self.scale)
x = self.group_norm(x)
```

Rationale: Conv2d and GroupNorm are mature ACL-covered operations. The optimized Triton work is limited to the custom bias/scale/sigmoid epilogue where fusion avoids extra GM round trips.
