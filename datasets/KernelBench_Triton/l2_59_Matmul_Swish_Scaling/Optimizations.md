# Optimizations

## 1. Simplified Swish sigmoid expression

**Before**
```python
z = tl.exp(-tl.abs(x))
s = tl.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
out = (x * s) * scale
```

**After**
```python
x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False).to(tl.float32)
sigmoid = 1.0 / (1.0 + tl.exp(-x))
out = x * sigmoid * scale
```

Rationale: the branch-stable form issues two vector-divide groups plus `abs`/`where` work. For Swish, the direct sigmoid form preserves the PyTorch reference within tolerance and cuts vector events in the default epilogue trace from 1640 to 998.

## 2. Explicit FP32 epilogue math

```python
x = tl.load(...).to(tl.float32)
...
tl.store(y_ptr + offs, out, mask=mask)
```

Rationale: computing the activation in FP32 matches `F.silu` closely for fp16 input. Remote correctness improved to `max_abs=5.96046e-08` for the optimized provider on all benchmark shapes.

## 3. Contiguous epilogue alignment hints and padding relaxation

```python
tl.multiple_of(offs, 16)
tl.max_contiguous(offs, BLOCK_SIZE)
x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
```

Rationale: the linear output is contiguous, so a flat 1D epilogue is the right access pattern. Alignment hints make the contiguous GM→UB/UB→GM pattern explicit; `care_padding=False` is safe because masked padding does not contribute to any reduction.

## 4. Grid-cap-safe direct/persistent dispatch

```python
n_tiles = triton.cdiv(n_elements, block_size)
if n_tiles > _MAX_PROGRAMS:
    _swish_scale_persistent[(_MAX_PROGRAMS,)](..., _MAX_PROGRAMS, BLOCK_SIZE=block_size)
else:
    _swish_scale_direct[(n_tiles,)](..., BLOCK_SIZE=block_size)
```

Rationale: normal shapes use the lower-overhead direct path, while very large outputs avoid Ascend's 65535-program FFTS cap. `profile_kernels.py` force-tests the persistent path by lowering `_MAX_PROGRAMS` to 1 on a modest shape.

## 5. Shape-aware block sizing retained

```python
if n_elements <= 262_144:
    block_size = 1024
elif n_elements <= (1 << 20):
    block_size = 4096
else:
    block_size = 8192
```

Rationale: the default KernelBench shape uses 8192-element tiles, matching the cannsim comparison. Smaller shapes keep smaller tiles to avoid the regression seen with a fixed 8192 block on small outputs.
