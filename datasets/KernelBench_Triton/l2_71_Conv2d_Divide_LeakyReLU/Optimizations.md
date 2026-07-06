# Optimizations

## 1. Fixed-size direct epilogue tile with grid-cap fallback

**Code**
```python
_BLOCK_SIZE = 8192
_MAX_GRID = 65535
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles <= _MAX_GRID:
    _div_leakyrelu_direct_kernel[(n_tiles,)](..., BLOCK_SIZE=_BLOCK_SIZE)
else:
    _div_leakyrelu_persistent_kernel[(_MAX_GRID,)](..., n_programs=_MAX_GRID, BLOCK_SIZE=_BLOCK_SIZE)
```

**Rationale**: The default problem has a large contiguous Conv2d output but still stays under the Ascend FFTS grid cap with 8192-element tiles. A separate persistent fallback preserves correctness for larger accepted inputs without making the common/default path pay a work-stealing loop.

## 2. Replaced autotune sweep with a UB-safe constant tile

**Code**
```python
@triton.jit
def _div_leakyrelu_direct_kernel(..., BLOCK_SIZE: tl.constexpr):
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

**Rationale**: The baseline autotune includes very large `BLOCK_SIZE` values up to 32768, which are risky for FP32 vector epilogues under UB pressure and increase compile/tuning overhead. `BLOCK_SIZE=8192` keeps one contiguous vector tile within the 192 KB UB envelope while halving the full-shape program count versus 4096.

## 3. Removed unsupported/low-value cache modifier and enabled padding-care elision

**Code**
```python
x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
tl.store(y_ptr + offs, out, mask=mask)
```

**Rationale**: Ascend Triton guidance warns that `cache_modifier=".cg"` is not useful on Ascend and can break compilation in some environments. Masked tail lanes are not consumed, so `care_padding=False` is safe for this pointwise epilogue.

## 4. Rewrote LeakyReLU algebra to branch-select form

**Before**
```python
y = x * inv
y_neg = tl.minimum(y, zero)
out = y + y_neg * (slope - one)
```

**After**
```python
y = x * inv
out = tl.where(y >= zero, y, y * slope)
```

**Rationale**: cannsim showed the baseline form emitted many `RV_VMINS`/extra vector instructions. The `tl.where` form reduced trace events from 1665 to 871 and wall cycles from 4287 to 3773 for the one-tile epilogue probe.
