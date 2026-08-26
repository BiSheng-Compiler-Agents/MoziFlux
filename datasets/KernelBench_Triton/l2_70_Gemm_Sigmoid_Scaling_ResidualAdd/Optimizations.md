# Optimizations Applied

## 1. Grid-cap-safe direct + persistent dispatch

The input launches one direct epilogue kernel for all sizes. The optimized host keeps the fast direct path for normal tensors and adds a persistent fallback only when the direct grid would exceed Ascend FFTS `coreDim <= 65535`.

```python
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_GRID:
    _sigmoid_scale_residual_persistent[(_MAX_GRID,)](..., n_programs=_MAX_GRID)
else:
    _sigmoid_scale_residual_direct[(n_tiles,)](...)
```

Rationale: the default shape (`1024 * 8192`) uses only 512 programs at `BLOCK_SIZE=16384`, so it stays on the direct path. Larger accepted shapes no longer risk grid overflow.

## 2. Stable large contiguous epilogue block

The optimized epilogue uses one contiguous 1D tile over the post-GEMM output and keeps the default large block size:

```python
_BLOCK_SIZE = 16384
x = self.gemm(x).contiguous()
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

Rationale: cannsim showed that reducing the tile to 4096 reduced raw sub-kernel cycles but regressed normalized cycles/element. The final optimized kernel keeps the 16384-element tile, which has the best normalized throughput for this streaming vector epilogue.

## 3. Removed cache/eviction hints from the optimized kernel

The optimized Triton kernel avoids the baseline's streaming cache modifiers:

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)
```

Rationale: Ascend kernels are sensitive to cache modifiers; avoiding non-essential hints keeps the kernel portable across triton-ascend compiler versions. cannsim shows the generated trace is effectively unchanged at the default block size, so this is a robustness optimization rather than a latency win.

## 4. `num_stages=2` for all optimized launches

The baseline uses `num_stages=1` for smaller tensors. The optimized code uses `num_stages=2` for both direct and persistent launches:

```python
_sigmoid_scale_residual_direct[(n_tiles,)](..., num_warps=4, num_stages=2)
```

Rationale: `num_stages=1` is an Ascend compiler/runtime pitfall. The default shape already used `num_stages=2`, but smaller accepted shapes are now safer.

## 5. Forced persistent-path test hook

`profile_kernels.py` lowers `_MAX_GRID` during unit testing to exercise the persistent path without allocating a >1B element tensor.

```python
mod._MAX_GRID = 1
# run modest shape and compare against PyTorch reference
```

Rationale: every dispatch path must be tested. Production dispatch still uses `_MAX_GRID = 65535`.
