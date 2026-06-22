# Optimizations

## 1. Avoid `movedim(...).contiguous()` for the target `dim=1` path

Baseline host path materializes `[B, N, M]` before reducing, which adds a full input-size GM copy before the kernel:

```python
x_last = x.movedim(dim, -1).contiguous()
```

Optimized dispatch keeps the original contiguous `[B, M, N]` layout and reduces the middle dimension in-place:

```python
if x.dim() == 3 and dim == 1:
    return _launch_dim1_3d(x)
```

This removes the dominant pre-kernel copy for the benchmark shape `(128, 4096, 4095)`.

## 2. Compute a tile of output columns per program

Baseline computes one output index per program and scans the reduction axis serially:

```python
best_val = tl.full((), float("inf"), dtype=tl.float32)
while k0 < cols:
    values = tl.load(x_ptr + row_base + offsets, mask=mask, other=float("inf"))
    tile_min = tl.min(values, axis=0)
```

Optimized code computes `BLOCK_N=64` output columns for one batch tile at once while streaming `BLOCK_M=64` rows of the middle dimension:

```python
vals = tl.load(x_ptr + b * M * N + m_idxs[:, None] * N + n_idxs[None, :], mask=mask, other=float("inf")).to(tl.float32)
tile_min = tl.min(vals, axis=0)
tile_idx = tl.min(tl.where(vals == tile_min[None, :], m_idxs[:, None], M), axis=0)
```

The optimized sub-kernel has slightly more cycles per program but produces 64 argmin outputs per program, reducing normalized cycles/output by about `56.3x` in cannsim.

## 3. Grid-capped persistent dispatch

All optimized kernels launch at most the physical vector core count and always below Ascend FFTS `coreDim <= 65535`:

```python
nprog = max(1, min(65535, _num_vector_cores(x), total_tiles))
_kernel[(nprog,)](..., total_tiles, nprog, ...)
```

Inside the kernel, each program loops over logical tiles:

```python
tile = tl.program_id(0)
while tile < total_tiles:
    ...
    tile += n_programs
```

This covers the exact target shape where the logical tile count exceeds physical core count while avoiding invalid large launches.

## 4. Separate fallback paths for other dimensions

The optimized `ModelNew` preserves the baseline interface and supports `dim=0`, `dim=1`, `dim=2`, negative dims, and non-3D fallback inputs:

```python
if x.dim() == 3 and dim == 0:
    return _launch_dim0_3d(x)
if dim == x.dim() - 1:
    return _launch_lastdim(x, out_shape)
return _launch_lastdim(x.movedim(dim, -1), out_shape)
```

The profiler unit test covers all 3D dispatch paths plus a 2D fallback case.
