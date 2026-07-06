# Optimizations Applied

## 1. Backend GEMM+bias dispatch via `F.linear`

```python
y = F.linear(x, weight, bias)
```

The input implementation used `torch.matmul(x, weight.transpose(0, 1))` and then a separate bias add. `F.linear` preserves the same math while giving the Ascend/PyTorch backend the canonical linear pattern and a contiguous epilogue input.

## 2. Retiled the post-GEMM epilogue from 1024 to 4096 contiguous elements

```python
_BLOCK_SIZE = 4096
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
_epilogue_direct[(n_tiles,)](..., BLOCK=_BLOCK_SIZE)
```

The GEMM output is contiguous, so a flat 1D epilogue is optimal. The target output has 8,388,608 elements: 4096-element tiles reduce direct epilogue programs from 8192 to 2048, cutting dispatch/loop overhead while preserving contiguous MTE access.

## 3. Added direct + persistent dispatch for Ascend FFTS grid cap

```python
if n_tiles > _MAX_PROGRAMS:
    _epilogue_persistent[(_MAX_PROGRAMS,)](x, out, n_elements, _MAX_PROGRAMS, BLOCK=_BLOCK_SIZE)
else:
    _epilogue_direct[(n_tiles,)](x, out, n_elements, BLOCK=_BLOCK_SIZE)
```

Direct launch is fastest below the 65,535-program cap; persistent work stealing is only used for shapes that would overflow the Ascend launch grid. `profile_kernels.py` force-tests this path by temporarily lowering `_MAX_PROGRAMS`.

## 4. Removed mathematically redundant final clamp

```python
e2y = tl.exp(2.0 * y)
y = (e2y - 1.0) / (e2y + 1.0)
tl.store(y_ptr + offsets, y.to(x.dtype), mask=mask)
```

The final operation in the problem is `clamp(tanh(...), -1, 1)`. Since tanh is already in `[-1, 1]`, the post-tanh min/max pair is redundant in fp32 and was removed to reduce vector work.

## 5. Used `care_padding=False` on masked contiguous loads

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
```

Padded lanes are masked off and do not contribute to any reduction or normalization. Disabling padding care is safe here and removes conservative padding handling from the vector load.
