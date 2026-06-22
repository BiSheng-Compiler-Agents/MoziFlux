# Optimizations for 35_GroupNorm_

## 1. Parallelized group statistics with per-group partial sums

Baseline computes one complete `(N, group)` reduction in a single program:

```python
stats_grid = (N * num_groups,)
while start < group_elems:
    x = tl.load(...)
    s += tl.sum(x)
    ss += tl.sum(x * x)
```

Optimized code splits each group into up to 32 independent partial reducers and finalizes with a short reduction kernel:

```python
num_parts = max(1, min(32, 65535 // max(1, total_groups)))
stats_grid = (total_groups * num_parts,)
_groupnorm_partial_stats_kernel[stats_grid](..., NUM_PARTS=num_parts)
_groupnorm_finalize_stats_kernel[(total_groups,)](..., group_elems_inv=1.0 / group_elems)
```

Rationale: the benchmark group has `8 * 512 * 512` elements, so one program per group serializes a large reduction. Partial reducers expose inter-core parallelism while keeping accumulators in vector registers and replacing divisions with a host-computed reciprocal.

## 2. Tensor accumulators instead of scalar loop-carried state

```python
acc = tl.zeros([1], dtype=tl.float32)
acc2 = tl.zeros([1], dtype=tl.float32)
acc += tl.sum(x, axis=0, keep_dims=True)
acc2 += tl.sum(x * x, axis=0, keep_dims=True)
```

Rationale: vector tensor accumulators avoid scalar spill/reload storms in multi-iteration reductions on Ascend.

## 3. Larger apply tile and capped dispatch fallback

Baseline normalizes with `BLOCK_HW=256`. Optimized direct channel apply uses `BLOCK_HW=1024`, reducing loop/control overhead per channel:

```python
for start in tl.range(0, HW, BLOCK_HW):
    idx = start + tl.arange(0, BLOCK_HW)
    x = tl.load(x_ptr + base + idx, mask=idx < HW, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + idx, ((x - mean) * rstd) * gamma + beta, mask=idx < HW)
```

For `N*C > 65535`, the optimized host switches to a persistent tile loop capped at Ascend's FFTS grid limit, and the profiler unit test covers this dispatch path with `dispatch_persistent`.
