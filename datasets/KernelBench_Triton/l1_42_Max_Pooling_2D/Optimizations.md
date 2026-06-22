# Optimizations Applied

## 1. Grid-capped persistent dispatch for Ascend legality

**Before**
```python
grid = (N * C, grid_ho * grid_wo)
```
The target shape launches `2048 * 512 = 1,048,576` logical programs, which exceeds Ascend's `coreDim <= 65535` launch limit.

**After**
```python
total_out = N * C * H_out * W_out
n_tiles = triton.cdiv(total_out, BLOCK)
grid = (min(n_tiles, _MAX_GRID),)
```
Inside the kernel:
```python
for tile_id in range(pid, n_tiles, n_programs):
    offs = tile_id * BLOCK + tl.arange(0, BLOCK)
```
This preserves the full output domain while capping the physical launch grid.

## 2. Flattened contiguous output tiling

**Before**
```python
pid_nc = tl.program_id(0)
pid_hw = tl.program_id(1)
HO = ho_offsets[:, None]
WO = wo_offsets[None, :]
```
The original 2D launch was illegal at target size and carried extra tile-shape control.

**After**
```python
wo = offs % W_out
tmp = offs // W_out
ho = tmp % H_out
c = (tmp // H_out) % C
n = (tmp // H_out) // C
```
A flat output vector keeps stores contiguous and makes persistent looping simple and general.

## 3. Fully masked generic pooling window

```python
in_bounds = mask & ih_in & iw_in
val = tl.load(x_ptr + row_base + safe_iw, mask=in_bounds, other=-float("inf"))
max_val = tl.maximum(max_val, val)
tl.store(y_ptr + offs, max_val, mask=mask)
```
All pooling parameters accepted by the baseline constructor are handled by one masked kernel path; no new shape guards were added.

## 4. Benchmark/profile coverage

`profile_kernels.py` covers small direct, generic 2x2, persistent medium, and the original target shape. Both read-only baselines are retained as comparison columns; their compile failures are reported as `INFO_EXCEPTION`, while optimized correctness is gated by `UNIT_TEST PASS`.
