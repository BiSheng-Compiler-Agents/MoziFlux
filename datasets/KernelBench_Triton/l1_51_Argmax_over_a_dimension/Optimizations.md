# Optimizations

## 1. Removed target-shape `movedim(...).contiguous()` copy

Baseline normalizes every reduction dimension by moving it to the last axis and materializing a contiguous copy:

```python
x_last = x.movedim(dim, -1).contiguous()
```

For the target `[128, 4096, 4095]`, `dim=1`, this creates an additional ~8.6 GB fp32 GM traffic before the Triton kernel.  The optimized path operates directly on contiguous `[B, M, N]` layout for `dim=1`:

```python
ptrs = x_ptr + b * M * N + offs_m[:, None] * N + offs_n[None, :]
```

## 2. Vectorized 2D tile reduction over the reduced dimension

Baseline computes one output element per program and streams the reduction axis serially:

```python
values = tl.load(x_ptr + row_base + offsets, mask=mask, other=-float("inf"))
tile_max = tl.max(values, axis=0)
```

The optimized kernel computes 128 adjacent output columns per program and reduces a `[64, 128]` tile with `tl.max(vals, axis=0)`:

```python
vals = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
tile_max = tl.max(vals, axis=0)
tile_idx = tl.min(tl.where(vals == tile_max[None, :], offs_m[:, None], M), axis=0)
```

This amortizes loop/control overhead and MTE setup across 128 output indices while preserving PyTorch first-index tie semantics.

## 3. Grid-capped persistent dispatch

The baseline launches `rows = B * N` programs after moving `dim=1` to the last dimension; target rows are 524,160, which exceeds Ascend's 65,535 grid limit.

```python
total_tiles = B * triton.cdiv(N, BLOCK_N)
grid = (min(total_tiles, 65535),)
for tile_id in tl.range(pid, total_tiles, tl.num_programs(0)):
    ...
```

The optimized launch remains legal for large shapes and each program processes multiple logical tiles when needed.

## 4. Correctness-preserving fallback for non-target dispatch paths

The optimized Triton path handles the target contiguous 3D `dim=1` regime. Other ranks/dimensions fall back to `torch.argmax(x, dim=dim)`, preserving the baseline interface without adding unsupported runtime guards.
