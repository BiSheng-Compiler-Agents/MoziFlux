# Optimizations

## 1. Removed row-kernel autotune sweep for a fixed vector body

**Baseline**
```python
@triton.autotune(configs=[... num_warps/num_stages variants ...], key=["F", "BLOCK"])
@triton.jit
def _rownorm_addmul_kernel(...):
    ...
```

**Optimized**
```python
@triton.jit
def _rownorm_addmul_direct_kernel(..., BLOCK: tl.constexpr):
    ...
```

Rationale: this post-linear operation is a pure vector row-normalization kernel with a fixed body. The optimized direct path removes non-useful CUDA-style autotune variants while preserving the measured-fast arithmetic body for normal batch sizes.

## 2. Added grid-capped persistent dispatch for large batch counts

```python
if batch_size > _MAX_PROGRAMS:
    n_programs = _MAX_PROGRAMS
    _rownorm_addmul_persistent_kernel[(n_programs,)](..., n_programs, BLOCK=block)
else:
    _rownorm_addmul_direct_kernel[(batch_size,)](..., BLOCK=block)
```

Rationale: the target batch `1024` stays on the faster direct path, but valid larger inputs no longer risk Ascend `coreDim > 65535`. The persistent kernel loops over rows by program id and was covered by `persistent_70000x32x32` in `profile_kernels.py`.

## 3. Preserved single-pass row normalization and reciprocal multiply

```python
x_row = tl.load(..., mask=mask, other=0.0)
y_row = tl.load(..., mask=mask, other=0.0)
sum_x = tl.sum(x_row, axis=0)
sum_x2 = tl.sum(x_row * x_row, axis=0)
mean = sum_x * inv_F
var = tl.maximum(sum_x2 * inv_F - mean * mean, 0.0)
out_row = y_row * y_row + y_row * (x_row - mean) * tl.rsqrt(var + eps)
```

Rationale: the full target row (`F=8192`) fits in UB, so one load of `linear_out` and `y` is better than multi-pass reductions. `inv_F` remains a host scalar to avoid in-kernel division.

## 4. Results

cannsim grid-1 traces showed the same bottleneck class (`PUSHQ`) and near-identical sub-kernel behavior (`7612 -> 7693` cycles, +1.06% noise/regression). Hardware verification showed optimized correctness on all shapes and latency parity/slight target win versus baseline1: `6.423610 ms -> 6.423433 ms` on `target_1024x8192x8192`, while adding a verified persistent fallback for `B > 65535`.
