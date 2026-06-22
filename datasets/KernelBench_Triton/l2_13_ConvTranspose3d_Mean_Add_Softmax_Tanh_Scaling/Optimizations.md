# Optimizations

## 1. Algebraic elimination of ConvTranspose3d + reductions

The operator contract is:

```python
y = conv_transpose3d(x)
y = y.mean(dim=1, keepdim=True)
y = y + bias
y = torch.softmax(y, dim=1)
y = torch.tanh(y) * scaling_factor
```

After `mean(dim=1, keepdim=True)`, the channel dimension is `1`; therefore `softmax(..., dim=1)` is exactly `1` for every spatial element, independent of convolution weights, input values, and bias. The optimized path keeps only the output-shape calculation and fills the result with `math.tanh(1.0) * scaling_factor`.

```python
out = torch.empty((n, 1, do, ho, wo), device=x.device, dtype=x.dtype)
_fill_const_direct[(n_tiles,)](out, self._const_val, n_elements, BLOCK_SIZE=8192)
```

Rationale: this removes the unused ConvTranspose3d, mean, add, softmax, tanh, and scaling work from the hot path while preserving the public constructor and output shape.

## 2. UB-safe constant fill tile

The editable input selected very large tiles up to `BLOCK_SIZE=131072` for the target shape. That creates oversized offset vectors and is not UB-safe on Ascend. The optimized fill uses `BLOCK_SIZE=8192`, which keeps one fp32 vector tile plus offsets/masks within UB headroom.

```python
_BLOCK_SIZE = 8192
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
```

Rationale: the cannsim one-core probe showed almost identical absolute wall cycles for 4096 vs 8192 elements, so doubling tile size nearly halves cycles per element.

## 3. Direct + persistent dispatch for grid legality

The target shape uses only 1024 tiles, so the direct path is fastest. A separate persistent path is retained for larger valid inputs whose tile count exceeds Ascend's 65,535 launch cap.

```python
if n_tiles > _MAX_PROGRAMS:
    _fill_const_persistent[(_MAX_PROGRAMS,)](
        out, self._const_val, n_elements, _MAX_PROGRAMS, BLOCK_SIZE=_BLOCK_SIZE
    )
else:
    _fill_const_direct[(n_tiles,)](
        out, self._const_val, n_elements, BLOCK_SIZE=_BLOCK_SIZE
    )
```

Rationale: direct launch avoids persistent-loop scalar overhead for normal shapes; persistent dispatch prevents `coreDim > 65535` for oversized outputs.

## 4. Constructor compatibility fix

The editable input's `get_init_inputs()` passes `scaling_factor` as the sixth positional argument, while its constructor names that position `bias_shape`. The optimized constructor accepts both forms:

```python
if isinstance(bias_shape, (int, float)):
    scaling_factor = float(bias_shape)
    bias_shape = (1, 1, 1, 1, 1)
```

Rationale: preserves KernelBench initialization without modifying the input or read-only reference files.
