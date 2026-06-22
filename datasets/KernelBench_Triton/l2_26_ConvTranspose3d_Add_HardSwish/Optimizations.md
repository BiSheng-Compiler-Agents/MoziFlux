# Optimizations

## 1. Preserve ACL ConvTranspose3d and optimize only the epilogue

The main transposed convolution is a mature ACL-covered operator and the source kernel already uses `nn.ConvTranspose3d`. The optimization keeps that path and changes only the custom `add + HardSwish` Triton epilogue.

```python
x = self.conv_transpose(x)
return _fused_add_hswish_mul(x, add_input)
```

Rationale: replacing ConvTranspose3d with scalar/vector Triton loops would be structurally worse; the failure mode is the post-conv elementwise launch grid.

## 2. Direct + persistent epilogue dispatch

The target output has `128 * 64 * 32 * 32 * 32 = 268,435,456` elements. A direct launch at the original fp32 block size exceeds Ascend's 65,535 grid cap, so the optimized host keeps the direct path for legal shapes and routes only oversized shapes to a persistent tile loop.

```python
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _fused_add_hswish_persistent_kernel[(_MAX_PROGRAMS,)](
        x_c, add_c, out, n_elements, _MAX_PROGRAMS, BLOCK=_BLOCK_SIZE
    )
else:
    _fused_add_hswish_direct_kernel[(n_tiles,)](
        x_c, add_c, out, n_elements, BLOCK=_BLOCK_SIZE
    )
```

Rationale: persistent dispatch is a legality/full-shape latency fix; applying it unconditionally would add loop/control overhead to small shapes.

## 3. Persistent loop iterates over tiles with int64 offsets

```python
n_tiles = tl.cdiv(n_elements, BLOCK)
for tile_id in range(pid, n_tiles, n_programs):
    offs = (tile_id * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offs < n_elements
```

Rationale: the work-stealing loop must step over tiles, not raw elements. `tl.int64` offsets protect large tensors whose byte span can exceed 2 GiB.

## 4. FP32 epilogue math and reciprocal multiply

```python
z = x.to(tl.float32) + add.to(tl.float32)
clipped = tl.minimum(tl.maximum(z + 3.0, 0.0), 6.0)
out = z * clipped * 0.16666666666666666
```

Rationale: this preserves HardSwish precision and replaces division by 6 with a constant multiply in the optimized kernel.
