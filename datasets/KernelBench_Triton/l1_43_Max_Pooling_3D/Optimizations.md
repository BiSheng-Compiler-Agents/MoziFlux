# Optimizations Applied: Max Pooling 3D

## 1. Ascend-safe 1D tile dispatch

The baseline launches `(N*C*outD*outH, ceil(outW / BLOCK_W))`. For the target shape `(16,32,128,128,128)` this creates more than 65,535 logical programs, which can exceed Ascend `coreDim` limits.

```python
total_tiles = N * C * outD * outH * triton.cdiv(outW, _BLOCK_W)
if total_tiles <= _MAX_PROGRAMS:
    _maxpool3d_tile_kernel[(total_tiles,)](...)
else:
    _maxpool3d_persistent_kernel[(_MAX_PROGRAMS,)](..., total_tiles, _MAX_PROGRAMS, ...)
```

Rationale: flattening row/W tiles into a 1D launch preserves contiguous W-vector work while making large outputs legal via a persistent grid-capped path.

## 2. Persistent grid-capped path for large 3D volumes

```python
pid = tl.program_id(0)
for tile_id in range(pid, total_tiles, n_programs):
    w_tile = tile_id % n_w_tiles
    row = tile_id // n_w_tiles
    ...
```

Rationale: the target shape has ~1.97M W tiles. The persistent path keeps `grid <= 65535` and loops over tiles inside each program instead of launching an illegal full grid.

## 3. Preserve contiguous W-vector pooling and FP32 max accumulation

```python
ow = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
acc = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
...
vals = tl.load(x_ptr + base_nc + base_zh + x, mask=m, other=-float("inf"))
acc = tl.maximum(acc, vals.to(tl.float32))
tl.store(y_ptr + out_idx, acc, mask=mask_ow)
```

Rationale: vectorizing across output W keeps global memory accesses coalesced for each pooling position and uses FP32 reduction semantics for fp16/fp32 inputs.

## 4. Hoist x-coordinate validity outside depth/height loops

```python
x = in_x0
for kw in tl.static_range(0, K_W):
    x_valid = (x >= 0) & (x < W)
    for kd in tl.static_range(0, K_D):
        ...
    x += dil_w
```

Rationale: `x_valid` is invariant over `K_D*K_H`; hoisting reduces scalar/control work while preserving the same masked load behavior at padded boundaries.
