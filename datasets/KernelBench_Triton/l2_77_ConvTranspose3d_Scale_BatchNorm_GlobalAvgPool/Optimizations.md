# Optimizations Applied

## 1. Replace full ConvTranspose3d materialization with global-sum algebra

For the default `ConvTranspose3d` geometry (`stride=padding=output_padding=0/1`, `dilation=1`, `groups=1`) in eval mode, the spatial sum of the transposed convolution can be computed without constructing the full `[N, C_out, D_out, H_out, W_out]` tensor:

```python
input_sums = _spatial_sum3d_triton(x)                         # [N, C_in]
weight_sums = self.conv_transpose.weight.to(torch.float32).sum(dim=(2, 3, 4))
y_sum = torch.matmul(input_sums, weight_sums) * float(self.scale_factor)
y_sum = y_sum + bias * (scale_factor * out_elems)
mean = y_sum * (1.0 / float(out_elems))
```

Rationale: the benchmark shape would otherwise materialize 53,084,160 convolution-output elements. The optimized path reads 16,777,216 input elements for the Triton spatial sum and performs only a tiny `[16,64] @ [64,128]` channel-mixing GEMM.

## 2. Commute eval BatchNorm after global average pooling

The input baseline already used the identity `Avg(BN(x)) == BN(Avg(x))` in eval mode. The optimized code keeps the identity but applies it to the algebraic mean:

```python
affine = torch.rsqrt(rv + bn.eps) * bn.weight.to(torch.float32).view(1, -1)
out = (mean - rm) * affine + bias
```

Rationale: this avoids BatchNorm over the full 5D output and applies only `[N, C_out]` affine work.

## 3. Retile the Triton work to input spatial sums and add grid-cap dispatch

The optimized Triton kernel reduces one contiguous input `(N,C_in)` plane per program and has direct plus persistent dispatch:

```python
if rows <= _MAX_GRID:
    _spatial_sum3d_direct_kernel[(rows,)](x_flat, sums, rows, L=L, BLOCK=_SUM_BLOCK)
else:
    _spatial_sum3d_persistent_kernel[(_MAX_GRID,)](x_flat, sums, rows, _MAX_GRID, L=L, BLOCK=_SUM_BLOCK)
```

Rationale: the direct path is fastest for normal shapes, while the persistent path keeps the fallback legal if `N*C_in` exceeds Ascend's 65,535 grid limit. `profile_kernels.py` force-tests this persistent path by temporarily lowering `_MAX_GRID`.

## 4. Safe fallback for non-default/training cases

The algebraic shortcut is only used when its preconditions hold; all other cases route to the original semantic order:

```python
if not _can_use_sum_shortcut(self, x):
    return self._fallback_forward(x)
```

Rationale: this avoids adding unsupported runtime constraints. Training-mode BatchNorm and non-default transposed-convolution geometry remain correct through the ACL fallback.
