# Optimizations

## 1. Preserve ACL ConvTranspose2d and optimize only the Triton epilogue

The main transposed convolution is already dispatched through `nn.ConvTranspose2d`, a mature ACL operator. The optimized file keeps that path unchanged and focuses on the elementwise Mish/Add/Hardtanh/Scale epilogue.

```python
x = self.conv_transpose(x)
return _fused_mish_add_hardtanh_scale(x, self.add_value, self.scale)
```

Rationale: rewriting ConvTranspose2d as scalar/vector Triton loops would not use the convolution engine effectively; the actual target issue is the huge epilogue launch over 536,870,912 output elements.

## 2. Direct + persistent epilogue dispatch

The baseline launches one Triton program per 4096 elements. At the default shape this requires `ceil(536870912 / 4096) = 131072` programs, which exceeds Ascend's 65,535 grid cap.

```python
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _mish_add_hardtanh_scale_persistent_kernel[(_MAX_PROGRAMS,)](..., n_programs=_MAX_PROGRAMS)
else:
    _mish_add_hardtanh_scale_direct_kernel[(n_tiles,)](...)
```

Rationale: direct launch remains fastest for legal small/medium tensors, while persistent launch is required for the target output.

## 3. Tile-loop persistent kernel with int64 offsets

```python
pid = tl.program_id(0)
n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
for tile_id in range(pid, n_tiles, n_programs):
    offs = (tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offs < n_elements
```

Rationale: the persistent path loops over tiles, not elements, and promotes offsets to int64 so the >2 GiB output byte span is addressed safely.

## 4. Safer Ascend launch metadata

```python
num_stages=2
```

Rationale: `num_stages=3` is not useful on Ascend for this vector epilogue; the optimized path uses the conservative Ascend-safe setting while keeping the same 4096-element UB tile.
