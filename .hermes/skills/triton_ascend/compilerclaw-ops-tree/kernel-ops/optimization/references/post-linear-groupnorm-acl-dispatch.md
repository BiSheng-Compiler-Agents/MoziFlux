# Post-linear GroupNorm ACL dispatch

Use this pattern for `Linear/GEMM -> GroupNorm -> activation -> add-to-self/scale` chains on Ascend when the custom Triton baseline computes GEMM in very narrow per-group/channel tiles.

## When to choose ACL/CANN production dispatch

1. If the Triton kernel launches one GEMM tile per `(batch tile, group)` and `BLOCK_N = C / groups` is small (for example 16), it repeatedly reloads the same `x` tile across groups and under-utilizes Cube.
2. Treat `F.linear`/`torch.matmul`, `F.group_norm`, and standard activations (`leaky_relu`, `silu`, `relu`, etc.) as first-class optimized providers before investing in a fully custom fused Triton GEMM+norm kernel.
3. Keep cannsim claims separate: cannsim validates custom Triton sub-kernels/fallback bodies, while production hardware latency may come from ACL/CANN library dispatch.

## Production host pattern

```python
z = F.linear(x.contiguous(), weight, bias)
y = F.group_norm(z, num_groups, gamma, beta, eps)
return F.leaky_relu(y, negative_slope=negative_slope) * 2.0  # add-to-self / sum pattern
```

Use the exact activation/scale required by the operator; the point is to route the standard pieces to CANN when the custom fused GEMM shape is pathological.

## Triton fallback pattern

If retaining a Triton GroupNorm epilogue for diagnostics or small shapes:

```python
group_block = max(1, min(8, G, 2048 // block_c))
group_tiles = triton.cdiv(G, group_block)
total_tiles = N * group_tiles
for off in range(0, total_tiles, 65535):
    chunk = min(65535, total_tiles - off)
    _groupnorm_epilogue[(chunk,)](..., total_tiles, off, GROUP_BLOCK=group_block, BLOCK_C=block_c)
```

Inside the kernel, batch independent groups per program and remap invalid lanes before pointer arithmetic reaches memory operations:

```python
valid = (row < N) & (g[:, None] < G) & (c[None, :] < Cg)
safe_ch = tl.where(valid, g[:, None] * Cg + c[None, :], 0)
z = tl.load(z_ptr + row * stride_zm + safe_ch * stride_zc, mask=valid, other=0.0).to(tl.float32)
mean = tl.sum(z, axis=1) / Cg
centered = z - mean[:, None]
var = tl.sum(centered * centered, axis=1) / Cg
```

## Profiling/reporting notes

- Unit-test both production and forced Triton fallback paths when the fallback remains in the optimized file.
- If baseline and optimized cannsim probes process different numbers of groups per program, report normalized cycles per group, not only raw wall cycles.
- In `performance_report.md`, state explicitly whether the hardware latency is ACL production dispatch and whether cannsim is for the fallback body only.
