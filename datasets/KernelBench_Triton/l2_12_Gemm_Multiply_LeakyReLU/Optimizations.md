# Optimizations Applied

## 1. Hybrid host dispatch: ACL for small/medium GEMM

```python
if M < 512 or K < 4096 or N < 4096:
    return F.leaky_relu(F.linear(x, weight, bias) * float(multiplier),
                        negative_slope=float(negative_slope))
```

Hardware verification showed Ascend's mature ACL linear path is faster for the smaller GEMM regimes in this operator. The optimized host interface routes those shapes to ACL while preserving the exact `Linear -> multiply -> LeakyReLU` math.

## 2. Preserve the autotuned Triton path for the large target GEMM

```python
_linear_mul_leaky_kernel_auto[grid](x_c, weight_c, bias_buf, y, M, N, K, ...)
```

The target shape `(M=1024, K=8192, N=8192)` is already well-served by the baseline Triton autotune set and is slightly faster than PyTorch / ACL. The optimized kernel keeps the autotuned Triton path for this regime instead of forcing ACL globally.

## 3. Persistent grid-cap fallback for oversized matrices

```python
if total_tiles <= _MAX_GRID:
    ...  # autotuned direct path
else:
    _linear_mul_leaky_kernel_persistent[(min(total_tiles, _MAX_GRID),)](...)
```

Ascend launches cannot exceed 65,535 blocks. The fallback loops over logical output tiles inside a capped 1D grid, preserving correctness for matrix shapes whose output tile count exceeds the hardware launch cap.

## 4. Persistent fallback uses matmul-safe compiler hints

```python
tl.load(..., other=0.0, care_padding=False)
al.compile_hint(a, "dot_pad_only_k")
al.compile_hint(b, "dot_pad_only_k")
```

For the fallback path, zero-padded masked elements do not affect GEMM, and `dot_pad_only_k` limits Cube padding work. This path is for legality/generalization, not the measured target fast path.
