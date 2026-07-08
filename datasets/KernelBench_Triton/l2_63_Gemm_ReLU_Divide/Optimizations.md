# Optimizations Applied

## 1. Flattened contiguous post-linear epilogue with larger tiles

The source already keeps the GEMM on `F.linear()` and applies the ReLU/divide epilogue to the contiguous GEMM output. The optimized version keeps that mathematically equivalent split, but retunes the epilogue from 1024 to 4096 contiguous elements per program to reduce launch/program overhead and improve normalized vector throughput.

```python
_EPILOGUE_BLOCK_SIZE = 4096
n_tiles = triton.cdiv(n_elements, _EPILOGUE_BLOCK_SIZE)
_relu_scale_direct_kernel[(n_tiles,)](y, n_elements, scale,
                                      BLOCK_SIZE=_EPILOGUE_BLOCK_SIZE)
```

Rationale: the epilogue is pure elementwise work over a contiguous output tensor of shape `[batch, out_features]`. A flat 1D tile avoids row/column index arithmetic and lets MTE loads/stores remain contiguous.

## 2. Replaced divide with reciprocal multiply

The baseline device epilogue performs a vector divide in each lane:

```python
x = tl.where(x > 0, x / divisor, 0.0)
```

The optimized host computes the reciprocal once and the kernel uses a vector multiply:

```python
return _launch_relu_scale_inplace(y, 1.0 / divisor)
# device:
y = tl.where(x > 0.0, x * scale, 0.0)
```

Rationale: `RV_VMULS` is cheaper than `RV_VDIV`; the divisor is scalar and invariant across the whole tensor, so reciprocal multiplication preserves results for the fixed module divisor while reducing vector ALU cost.

## 3. Added direct + persistent dispatch

```python
if n_tiles > _MAX_PROGRAMS:
    _relu_scale_persistent_kernel[(_MAX_PROGRAMS,)](
        y, n_elements, _MAX_PROGRAMS, scale, BLOCK_SIZE=_EPILOGUE_BLOCK_SIZE)
else:
    _relu_scale_direct_kernel[(n_tiles,)](
        y, n_elements, scale, BLOCK_SIZE=_EPILOGUE_BLOCK_SIZE)
```

Rationale: direct launch is fastest for normal shapes. The persistent fallback only activates when `cdiv(n_elements, BLOCK_SIZE) > 65535`, avoiding Ascend FFTS grid overflow without penalizing the benchmark shape.

## 4. Added contiguous/alignment hints and padding-care bypass

```python
tl.multiple_of(offsets, 16)
tl.max_contiguous(offsets, BLOCK_SIZE)
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
```

Rationale: offsets are linear and masked, so the compiler can use contiguous memory transactions. Masked padding lanes do not contribute to stores, so `care_padding=False` removes unnecessary padding checks.
