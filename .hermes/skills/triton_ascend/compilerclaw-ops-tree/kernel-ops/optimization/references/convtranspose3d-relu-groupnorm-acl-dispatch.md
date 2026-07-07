# ConvTranspose3d + ReLU + GroupNorm ACL Dispatch

Use this reference when optimizing KernelBench-style models that run `nn.ConvTranspose3d` followed by ReLU and GroupNorm, especially when the editable provider fuses ReLU+GroupNorm in a custom Triton reduction kernel.

## Recognition pattern

- `forward()` calls `self.conv_transpose(x)` using `nn.ConvTranspose3d` / ACL.
- A custom Triton epilogue computes ReLU group statistics, then reloads the same group data to normalize and apply affine parameters.
- The epilogue has nested per-group/per-channel loops, scalar index math, repeated GM reads, and MTE/PUSHQ/SCALAR stalls in cannsim.
- Default 3D target shapes make the custom comparison provider slow or toxic to benchmark at full scale.

## Preferred production optimization

Keep convolution on ACL and route the standard post chain to native NPU operators:

```python
y = self.conv_transpose(x)
y = torch.relu(y)
return torch.nn.functional.group_norm(
    y,
    self.group_norm.num_groups,
    self.group_norm.weight,
    self.group_norm.bias,
    self.group_norm.eps,
)
```

Rationale: ReLU and GroupNorm are standard operators. This avoids bespoke two-pass group reductions, repeated ReLU recomputation, scalar-heavy channel loops, and large custom Triton launch/runtime risk.

## Optional Triton fallback

If retaining custom Triton code for coverage or future A/B testing, keep it disabled by default and restrict it to simple ReLU, not GroupNorm reduction:

```python
n_tiles = triton.cdiv(n_elements, BLOCK_SIZE)
if n_tiles > _MAX_GRID:
    _relu_persistent_kernel[(n_programs,)](...)
else:
    _relu_direct_kernel[(n_tiles,)](...)
```

Requirements:
- Direct path for `n_tiles <= 65535`; persistent path only above the grid cap.
- Persistent loop iterates over tiles, not elements.
- Unit tests must force both direct and persistent fallback paths by temporarily disabling ACL dispatch and lowering `_MAX_GRID` on a modest tensor.

## Cannsim and reporting

- Simulate the original custom epilogue with a small sub-kernel host to document its bottleneck.
- Simulate the optimized Triton fallback separately if present; do not claim cannsim measures the ACL production path directly.
- Report production speed from `remote_verify` / hardware, and keep cannsim as custom-kernel bottleneck evidence.

Example trace shape from this pattern: original ReLU+GroupNorm epilogue showed MTE2/VEC/MTE3/PUSHQ/SCALAR pressure; a simple ReLU fallback had far fewer events and lower wall cycles, while production ACL dispatch won on target hardware.

## Profiling notes

- Preserve constructor and module creation order so `torch.manual_seed(0)` weight matching remains valid.
- Keep `Baseline Triton2` parser-visible even when sandbox rules forbid reading `base_*.py`: print `TEST Baseline Triton2 <label>: SKIP_UNAVAILABLE ... max_abs=inf` and emit `inf` benchmark cells.
- If the full target Baseline Triton1 comparison is too slow/toxic, pre-skip that comparison timing only; still run optimized correctness and optimized target benchmark.
