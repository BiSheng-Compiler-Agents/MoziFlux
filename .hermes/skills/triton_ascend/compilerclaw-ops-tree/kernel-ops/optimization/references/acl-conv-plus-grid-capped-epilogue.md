# ACL Convolution with Grid-Capped Custom Epilogue

Use this when a baseline already dispatches the main convolution/transposed-convolution to ACL/PyTorch, but applies a custom Triton elementwise epilogue over the large output tensor.

## Recognition pattern

- `forward()` or helper calls `torch.nn.functional.conv*` / `conv_transpose*` for the main operator.
- A following Triton epilogue performs elementwise work such as clamp, scale, divide, bias, or activation; this includes exp/log-heavy fused activations such as Mish followed by add/clamp/scale.
- The full output tensor is huge enough that `ceil(numel / BLOCK_SIZE) > 65535`, so the direct epilogue launch is illegal on Ascend even though the ACL convolution itself is valid.

## Preferred optimization

Keep the ACL convolution path and fix only the epilogue dispatch:

```python
_MAX_PROGRAMS = 65535
_BLOCK_SIZE = 8192

n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _epilogue_persistent_kernel[(_MAX_PROGRAMS,)](
        y, n_elements, _MAX_PROGRAMS, ..., BLOCK_SIZE=_BLOCK_SIZE,
    )
else:
    _epilogue_direct_kernel[(n_tiles,)](
        y, n_elements, ..., BLOCK_SIZE=_BLOCK_SIZE,
    )
```

Inside the persistent kernel, loop over **tiles**, not elements:

```python
pid = tl.program_id(0)
n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
for tile_id in range(pid, n_tiles, n_programs):
    offsets = (tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offsets < n_elements
    x = tl.load(ptr + offsets, mask=mask, other=0.0)
    tl.store(ptr + offsets, epilogue(x), mask=mask)
```

Use `tl.int64` offsets when the output byte span can exceed ~2 GiB. For scalar divide by a module parameter, compute `inv_divisor = 1.0 / divisor` on the host and multiply in the kernel.

## Plane-tiled channelwise bias/scale epilogues

For Conv2d outputs shaped `[N, C, H, W]`, a flat `numel` tile often recomputes the channel for every element:

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
c_idx = (offs // hw) % channels
bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
```

When the epilogue is channelwise (`min/clamp + bias[c] + scale`, affine, etc.), prefer a plane/HW-tiled direct path: map each program to one contiguous `(N, C)` plane tile, load `bias[c]` once as a scalar, and use contiguous HW offsets for data movement.

```python
n_hw_tiles = tl.cdiv(hw, BLOCK_HW)
plane = pid // n_hw_tiles
hw_tile = pid - plane * n_hw_tiles
hw_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
x_offsets = plane * hw + hw_offsets
c_idx = plane % channels
bias = tl.load(bias_ptr + c_idx, mask=pid < total_tiles, other=0.0)
```

Compute dispatch in tiles, not elements: `total_tiles = N * C * cdiv(H*W, BLOCK_HW)`. Keep the direct path when `total_tiles <= 65535`; route only larger valid shapes to a persistent tile loop. Hardware can improve substantially even when a scalar cannsim microprobe looks worse, because the production win is fewer launches and less per-element div/mod/gather overhead.

## Fused activation epilogues

For epilogues like `Mish -> add -> Hardtanh -> scale`, keep the activation fused in Triton rather than decomposing into separate ACL/PyTorch elementwise calls unless hardware profiling proves the decomposition wins. A stable Mish implementation that avoids `tl.tanh` is:

```python
x_f32 = x.to(tl.float32)
sp = tl.maximum(x_f32, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x_f32)))
tanh_sp = 2.0 / (1.0 + tl.exp(-(2.0 * sp))) - 1.0
out = x_f32 * tanh_sp
out = tl.minimum(tl.maximum(out + add_value, -1.0), 1.0) * scale_value
```

For HardSwish epilogues, prefer the algebraic form in both Triton and the PyTorch reference when ACL `F.hardswish` is unavailable:

```python
z = x_f32 + add_f32
out = z * tl.minimum(tl.maximum(z + 3.0, 0.0), 6.0) * 0.16666666666666666
# PyTorch reference: z * torch.clamp(z + 3.0, 0.0, 6.0) / 6.0
```

When the direct baseline exceeds the grid cap at the target shape, the cannsim sub-kernel may show the persistent path slightly slower per tile because of loop/control overhead. Treat this as expected: the optimization is dispatch legality and full-shape hardware latency, not per-tile instruction reduction. Keep the direct path for legal small/medium shapes.

## Profiling/reporting notes

- cannsim grid=1 traces for direct vs persistent epilogues should be near-identical; the benefit is full-shape launch legality, not per-tile instruction reduction.
- Report baseline target latency as `inf` / `grid_guard` when the direct epilogue would exceed 65,535 programs.
- Keep all provider columns visible in `profile_kernels.py`, and require optimized correctness on the target shape.
