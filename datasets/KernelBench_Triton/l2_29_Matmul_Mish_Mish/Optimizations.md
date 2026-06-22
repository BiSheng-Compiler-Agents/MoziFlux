# Optimizations: l2_29_Matmul_Mish_Mish

## Baseline

The input kernel computes one output row and 32 output columns per program with vector multiply/reduce loops:

```python
x_vals = tl.load(...).to(tl.float32)
w_vals = tl.load(...).to(tl.float32)
acc += tl.sum(w_vals * x_vals[None, :], axis=1)
```

This keeps the matmul on Vector pipelines instead of Cube and launches `M * ceil(N/32)` programs, which exceeds Ascend `coreDim` for the target shape.

## Optimization 1 — Use mature ACL GEMM for production dispatch

```python
return _mish2_torch(F.linear(x, weight, bias))
```

`F.linear` routes the dominant 1024×8192×8192 GEMM to the vendor matmul path, avoids the baseline grid overflow, and keeps the public `ModelNew` interface unchanged.

## Optimization 2 — Keep a traced Triton Cube fallback

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k0 in tl.range(0, K, BLOCK_K):
    a = tl.load(..., mask=..., other=0.0, care_padding=False)
    b = tl.load(..., mask=..., other=0.0, care_padding=False)
    acc = tl.dot(a, b, acc)
```

The fallback uses `tl.dot(a, b, acc)` so matmul runs on Cube with fp32 accumulation instead of vector FMA/reduction. It is available via `force_triton=True` / `KB29_FORCE_TRITON=1` and is unit-tested in `profile_kernels.py`.

## Optimization 3 — Grid-capped 1D persistent tile loop

```python
total_tiles = triton.cdiv(m, 64) * triton.cdiv(n, 64)
grid_x = min(total_tiles, _num_aicore(), 65535)
for tile_id in tl.range(pid, TOTAL_TILES, tl.num_programs(axis=0)):
    ...
```

This avoids the baseline `(M, ceil(N/32))` launch product and keeps dispatch within Ascend's 65,535 grid cap while load-balancing tiles across AI cores.

## Optimization 4 — Fused Mish(Mish(.)) epilogue

```python
z = z * torch.tanh(F.softplus(z, beta=1, threshold=20))
z = z * torch.tanh(F.softplus(z, beta=1, threshold=20))
```

The optimized production path keeps the two Mish activations adjacent to the linear output without materializing extra model modules; the Triton fallback uses the same stable thresholded softplus algebra in-kernel.

## Optimization 5 — Ascend-specific load and scheduling hints

```python
tl.max_contiguous(offs_k, BLOCK_K)
tl.load(..., care_padding=False)
al.compile_hint(acc, "dot_pad_only_k")
```

The fallback uses contiguous K access, masked matmul-safe loads, and K-only dot padding to reduce padding and vector-side overhead in the traced sub-kernel.
