# Row-wise cumprod prefix-scan pattern

## Problem

A direct row-wise cumulative product implemented as a scalar `while i < N` loop emits one load, multiply, store, branch, and scalar bookkeeping sequence per element. On Ascend this creates heavy MTE3/PUSHQ/SCALARLDST pressure and can make even a single-row cannsim probe slow or unsafe at large `N`.

## Preferred pattern

Use a vector prefix scan inside fixed-size chunks and keep only a scalar carry between chunks:

```python
@triton.jit
def _cumprod_rowwise_block_kernel(x_ptr, y_ptr, M, N, sxm, sxn, sym, syn, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    rel = tl.arange(0, BLOCK)
    x_row = x_ptr + row * sxm
    y_row = y_ptr + row * sym
    carry = tl.full((), 1.0, dtype=tl.float32)

    for start in tl.range(0, N, BLOCK):
        offs = start + rel
        mask = offs < N
        vals = tl.load(x_row + offs * sxn, mask=mask, other=1.0).to(tl.float32)
        scan = tl.cumprod(vals, axis=0)
        tl.store(y_row + offs * syn, scan * carry, mask=mask)

        valid = tl.minimum(BLOCK, N - start)
        last_rel = valid - 1
        block_last = tl.sum(tl.where(rel == last_rel, scan, 0.0), axis=0)
        carry *= block_last
```

Correctness notes:
- `other=1.0` is the neutral element for masked tail lanes.
- The scalar `carry` must multiply the current chunk output before being updated with that chunk's last prefix product.
- Use fp32 scan/carry for floating-point inputs, then store through the output pointer dtype.
- Preserve generic `dim` support by moving the scan dimension to the last axis, flattening rows, then inverse-permuting.

## Grid-cap variant

If row count can exceed Ascend's 65,535 launch-program cap, keep the direct path for `rows <= 65535` and add a persistent row loop for larger inputs:

```python
@triton.jit
def _cumprod_rowwise_block_persistent_kernel(..., n_programs, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    while row < M:
        # complete one row using the same block-prefix scan
        row += n_programs
```

Route with `n_programs = min(rows, 65535)`. The persistent path is a legality/generalization fix; full-shape hardware timing is needed to measure dispatch benefit.

## Session data point

For KernelBench `l1_90_cumprod` (`32768 x 32768`, `dim=1`), replacing the scalar loop with `BLOCK=512` chunked `tl.cumprod` produced:

| Measurement | Baseline scalar loop | Block prefix scan |
|---|---:|---:|
| Cannsim bounded probe (`N=64`) wall cycles | 18,934 | 5,037 |
| Required-shape hardware latency | 2,917.264648 ms | 306.718933 ms |

The read-only golden `base_90_cumprod.py` was slightly faster at 299.259674 ms, so this pattern is a strong baseline fix but still worth comparing against any existing optimized/golden implementation.
