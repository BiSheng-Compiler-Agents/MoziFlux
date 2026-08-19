# Optimizations Applied

## 1. Larger activation tile for default Conv2d output

**Change:** the Mish activation Triton tile was increased from 4096 to 8192 elements.

```python
_MAX_PROGRAMS = 65535
_BLOCK_SIZE = 8192
```

**Rationale:** the default convolution output has 130,056,192 activation elements. A larger tile halves Triton program count and amortizes scalar setup, queue push, and MTE wait overhead while staying within UB for the elementwise fp32 Mish formula. Cannsim confirms the 8192-element tile processes twice as many values with only 4817 wall cycles vs 3952 cycles for the 4096 baseline tile, reducing normalized cycles per 4096 elements from 3952 to 2408.5.

## 2. Direct + persistent dispatch guard

**Change:** the host interface now uses the direct one-program-per-tile launch for normal sizes and a separate persistent kernel only when tile count exceeds the Ascend grid cap.

```python
n_tiles = triton.cdiv(n, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _mish_persistent_kernel[(_MAX_PROGRAMS,)](xi, y, n, _MAX_PROGRAMS, ...)
else:
    _mish_direct_kernel[(n_tiles,)](xi, y, n, ...)
```

**Rationale:** persistent dispatch avoids illegal/oversized FFTS grids for future larger tensors without penalizing the default shape, which remains below the threshold with `BLOCK_SIZE=8192`. `profile_kernels.py` includes a forced-persistent unit test to cover this dispatch path without allocating a huge tensor.

## 3. Preserve stable one-exp Mish formula

**Change:** retained the baseline's algebraic `x * tanh(softplus(x))` implementation using one exponentiation and a threshold branch.

```python
t = tl.exp(tl.where(use_large, 0.0, x))
den = t * t + 2.0 * t + 2.0
tanh_sp = tl.where(use_large, 1.0, 1.0 - 2.0 / den)
y = (x * tanh_sp).to(x_in.dtype)
```

**Rationale:** this avoids unavailable/expensive `tl.tanh`/`tl.log` paths on Triton-Ascend while matching PyTorch `softplus(beta=1, threshold=20)` within fp32 tolerances. Remote correctness max absolute error was <= 1.78814e-07 on all benchmark shapes.
