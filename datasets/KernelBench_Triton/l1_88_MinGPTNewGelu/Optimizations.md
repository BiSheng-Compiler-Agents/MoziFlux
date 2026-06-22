# Optimizations Applied

## 1. 2D row-major block-pointer tiling

Baseline uses a flat 1D pointer tile:
```python
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
```
Optimized code treats the tensor as `[rows, cols]` and uses row-major block pointers:
```python
x_block_ptr = tl.make_block_ptr(
    base=x_ptr, shape=(rows, cols), strides=(row_stride, 1),
    offsets=(row_start, 0), block_shape=(4, 2048), order=(1, 0)
)
x = tl.load(x_block_ptr)
```
Rationale: the target input is 2D `(8192, 8192)`, so a `4 x 2048` tile doubles useful work per program while preserving contiguous GM access.

## 2. Mask-free even path plus masked fallback

```python
kernel = _gelu_fwd_kernel_even if even else _gelu_fwd_kernel
```
For shapes divisible by `BLOCK_ROWS=4` and `BLOCK_COLS=2048`, the even path removes predicate/boundary overhead; non-divisible shapes still use `boundary_check=(0, 1)` for correctness.

## 3. Grid-capped persistent row-tile path

```python
if row_tiles > _MAX_PROGRAMS:
    _gelu_fwd_kernel_persistent[(65535,)](...)
```
Ascend launch grids must not exceed 65,535 blocks. The optimized host keeps the fast direct path for normal shapes and routes only oversized row-tile counts to a persistent loop.

## 4. Preserved minGPT approximate GELU math

```python
u = x32 * (0.7978845608028654 + 0.035677408136300125 * x32 * x32)
y = 0.5 * x32 * (1.0 + tl.math.tanh(u))
```
The computation remains the OpenAI GPT/minGPT approximate GELU formula, with FP32 intermediate math and output cast back to the input dtype.
