# Optimizations

## 1. Plane-tiled fused epilogue

Baseline used a flat element tile and recomputed the channel for every element:

```python
c_idx = (offs // hw) % channels
bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
```

The optimized kernel maps each program to one contiguous `(N, C)` output plane tile, so channel/bias selection is scalar per tile while loads and stores remain contiguous:

```python
n_hw_tiles = tl.cdiv(hw, BLOCK_HW)
plane = pid // n_hw_tiles
hw_tile = pid - plane * n_hw_tiles
hw_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
x_offsets = plane * hw + hw_offsets
c_idx = plane % channels
bias = tl.load(bias_ptr + c_idx, mask=pid < total_tiles, other=0.0)
```

Rationale: removes per-element integer division/modulo from the hot vector offsets and changes bias access from a vector gather to a scalar plane load.

## 2. Larger HW tile for fewer launches

Baseline `BLOCK_SIZE=4096` produces `ceil(numel/4096)=63,504` programs at the exact shape. The optimized direct path uses `BLOCK_HW=8192`, so the exact shape launches `N*C*ceil(HW/8192)=32,768` programs.

```python
_BLOCK_HW = 8192
n_hw_tiles = triton.cdiv(hw, _BLOCK_HW)
total_tiles = n_planes * n_hw_tiles
```

Rationale: the epilogue is memory/launch overhead dominated, and 8192 fp32 elements fit within UB headroom for the live load/compute/store vectors.

## 3. Direct + persistent dispatch for grid safety

The optimized host keeps the fastest direct launch below the Ascend 65,535 program cap and routes only oversized valid shapes to a persistent tile loop:

```python
if total_tiles > _MAX_PROGRAMS:
    _min_bias_scale_persistent_kernel[(_MAX_PROGRAMS,)](..., total_tiles, ..., _MAX_PROGRAMS)
else:
    _min_bias_scale_direct_kernel[(total_tiles,)](..., total_tiles, ...)
```

Rationale: the exact benchmark remains direct and faster, while larger batch/spatial regimes avoid `coreDim > 65535` without adding persistent-loop overhead to normal shapes.

## 4. Kept ACL Conv2d unchanged

The convolution remains `nn.Conv2d`/ACL-backed:

```python
x = self.conv(x)
return _apply_min_bias_scale(x, self.bias, self.constant_value, self.scaling_factor)
```

Rationale: the main operator is mature ACL convolution; the optimized work is limited to the custom fused epilogue that the baseline already used.
