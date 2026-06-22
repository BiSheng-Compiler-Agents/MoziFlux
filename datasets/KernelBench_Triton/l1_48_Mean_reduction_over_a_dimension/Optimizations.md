# Optimizations

## 1. Replaced scalar M-loop dim=1 reduction with 2D block reduction

Baseline reduced one contiguous N vector for every scalar `m` step:

```python
while m < M:
    vals = tl.load(x_ptr + b * stride_b + mi * stride_m + offs_n * stride_n, ...)
    acc += vals.to(tl.float32)
```

The optimized kernel loads a `[BLOCK_M, BLOCK_N]` tile and reduces the M axis in UB:

```python
for m0 in tl.range(0, M, BLOCK_M):
    vals = tl.load(x_ptr + b * stride_b + m_idxs[:, None] * stride_m + offs_n[None, :] * stride_n,
                   mask=(m_idxs[:, None] < M) & n_mask[None, :], other=0.0).to(tl.float32)
    acc += tl.sum(vals, axis=0)
```

Rationale: the target reduces `M=4096` over `N=4095`; block reduction cuts loop/control work from one step per row to one step per 128 rows while preserving contiguous loads over N.

## 2. Grid-capped persistent dispatch for oversized grids

All optimized paths launch a 1D grid capped at Ascend's 65,535 program limit and stride over remaining work in-kernel:

```python
n_programs = min(total_tiles, _MAX_GRID)
_kernel[(n_programs,)](..., total_tiles, n_programs, ...)
```

```python
tile = tl.program_id(0)
while tile < total_tiles:
    ...
    tile += n_programs
```

Rationale: baseline `dim=2` launches `B*M` programs, which overflows for the provided `128*4096` shape; the optimized path remains legal for all three reduction dimensions.

## 3. Correct ACL fallback for non-target dimensions

The benchmark problem constructs `ModelNew(dim=1)`, so the optimized Triton path targets `dim=1`. Non-target dimensions are delegated to `torch.mean` after preserving the baseline validation contract:

```python
if dim != 1:
    return torch.mean(x, dim=dim)
```

Rationale: this keeps `dim=0` and `dim=2` correct and avoids Ascend `coreDim` overflow in the read-only comparison kernels while preserving the optimized target dispatch.

## 4. Ascend-friendly launch metadata and alignment hints

The optimized kernels use `num_stages=2` and add contiguity/alignment hints on N offsets:

```python
offs_n_base = tl.max_contiguous(tl.multiple_of(offs_n_base, 16), BLOCK_N)
```

Rationale: `num_stages=4` is unnecessary on Ascend; alignment hints help MTE generate wider contiguous transfers for the innermost N dimension.
