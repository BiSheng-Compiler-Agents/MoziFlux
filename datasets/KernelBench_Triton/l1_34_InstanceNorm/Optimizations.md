# Optimizations Applied

## 1. Two-path host dispatch

```python
if total_tiles <= _MAX_PROGRAMS:
    _instancenorm_single_kernel[(total_planes,)](...)
    return y
```

Rationale: for small/medium planes the original one-program-per-(N,C) strategy avoids multi-launch overhead and stays under the Ascend FFTS `65535` program limit.

## 2. Persistent tiled path for oversized grids

```python
for tile in range(pid, TOTAL_TILES, n_programs):
    plane = tile // NUM_BLOCKS
    block = tile - plane * NUM_BLOCKS
```

Rationale: the source target shape has `112*64*ceil(512*512/4096)=458752` HW tiles, which exceeds the grid limit. The optimized path caps the launch at `65535` programs and uses grid-stride tile processing.

## 3. Three-stage large-HW normalization

```python
partial_sum = torch.empty((total_planes, num_blocks), device=x.device, dtype=torch.float32)
scale = torch.empty((total_planes,), device=x.device, dtype=torch.float32)
shift = torch.empty((total_planes,), device=x.device, dtype=torch.float32)
```

Rationale: for oversized planes, reduction is split into parallel partial sums, per-plane scale/shift finalization, then tiled apply. This exposes HW-block parallelism instead of serializing an entire 512x512 plane inside one program.

## 4. FP32 reduction and masked contiguous loads

```python
vals = tl.load(x_ptr + plane * HW + offs, mask=mask, other=0.0).to(tl.float32)
ss = tl.sum(vals, axis=0)
sq = tl.sum(vals * vals, axis=0)
```

Rationale: reductions remain FP32 for numerical stability; every load/store is masked for Ascend OOB safety, and offsets are contiguous to preserve DMA coalescing.
