# Optimizations

## 1. Dim=1 reduction changed from scalar row streaming to 2D block reduction

Baseline reduced one row at a time and executed `BLOCK_K` scalar-style row loads inside a static loop:

```python
for kk in tl.static_range(0, BLOCK_K):
    vals = tl.load(x_ptr + base_b + k_idx * N + n_offsets, mask=mask_n & (k_idx < M), other=0.0)
    acc += vals.to(tl.float32)
```

Optimized kernel loads a contiguous `[BLOCK_M, BLOCK_N]` tile and reduces the M axis in one vectorized operation:

```python
vals = tl.load(ptrs, mask=(m[:, None] < M) & mask_n[None, :], other=0.0).to(tl.float32)
acc += tl.sum(vals, axis=0)
```

Rationale: the target reduces `M=4096` for each `(B, N)` tile. Vectorizing across 128 M rows cuts loop/control instructions and reduces MTE2 transactions while preserving fp32 accumulation.

## 2. Persistent 1D grid capped by vector-core count

Optimized dispatch launches at most the physical vector-core count and loops over logical output tiles in-kernel:

```python
n_programs = max(1, min(cores, total_tiles, 65535))
_sum_dim1_kernel[(n_programs,)](..., total_tiles, n_programs, BLOCK_M=128, BLOCK_N=128)
```

```python
for tile in tl.range(pid, total_tiles, n_programs):
    ...
```

Rationale: the baseline launches one program per `(B, N tile)` and can create thousands of programs. The persistent loop keeps FFTS/coreDim legal, reduces dispatch pressure, and still covers all tiles.

## 3. Mask-complete kernels for all reduction dimensions

All optimized loads and stores use explicit masks:

```python
tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
tl.store(out_ptr + ..., acc, mask=mask)
```

Rationale: Ascend has zero tolerance for out-of-bounds accesses. The baseline had maskless fast paths for dim=0 and dim=2; the optimized implementation removes those paths and keeps boundary behavior correct for irregular shapes.

## 4. Smaller UB-safe dim=0 tile

Dim=0 uses `BLOCK_B=8, BLOCK_M=16, BLOCK_N=64` instead of the large 3D tile shape in the input kernel.

```python
_sum_dim0_kernel(..., BLOCK_B=8, BLOCK_M=16, BLOCK_N=64)
```

Rationale: a large `[B, M, N]` tile can exceed the 192 KB UB once fp32 intermediates, masks, and offsets are included. The smaller tile keeps the dim=0 fallback general and UB-safe.
