# Optimizations

## 1. Widened K tile: `BLOCK_K = 128 -> 512`

Baseline processes the very long reduction dimension in 128-wide chunks:

```python
BLOCK_M = 64
BLOCK_K = 128
```

Optimized code widens the intra-core reduction tile to 512:

```python
BLOCK_M = 64
BLOCK_K = 512
for k0 in tl.range(0, K, BLOCK_K):
    ...
```

Rationale: the benchmark has `K = 1,048,576`, so this cuts loop iterations and repeated MTE/Vector wait points by 4x while keeping the fp32 case within UB budget (`64 * 512 * 4 = 128 KiB` for the A tile, plus B/accumulator overhead).

## 2. `tl.range` loop instead of `while`

Baseline uses a loop-carried Python-style counter:

```python
k0 = 0
while k0 < K:
    ...
    k0 += BLOCK_K
```

Optimized code uses a structured Triton range:

```python
for k0 in tl.range(0, K, BLOCK_K):
    ...
```

Rationale: `tl.range` exposes the loop trip structure to the Ascend compiler and reduces scalar/control-flow overhead in long reductions.

## 3. `care_padding=False` on masked loads

```python
a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0, care_padding=False)
b_tile = tl.load(b_ptrs, mask=mask_k, other=0.0, care_padding=False)
```

Rationale: padding contributes zero to the dot product, so disabling padding care removes unnecessary load-side checks.

## 4. Physical vector-core capped grid with intra-core loop

```python
tile_count_m = triton.cdiv(M, BLOCK_M)
num_vectorcore = props.get("num_vectorcore", tile_count_m)
grid = (max(1, min(tile_count_m, num_vectorcore)),)
```

Rationale: avoids oversubscribing FFTS while preserving coverage for arbitrary `M`; each program handles `tile_m = pid, pid + nprog, ...`.

## Not adopted: Cube GEMV (`tl.dot`) path

A padded `N=16` Cube GEMV path was implemented and simulated, but it regressed for the `N=1` vector case: sub-kernel K=128 trace was `3443` cycles baseline vs `11009` cycles Cube path, and a K=512 Cube tile overflowed UB (`requires 2359296 bits while 2031616 bits available`). The final optimized kernel keeps vector reduction because cannsim showed it is faster for true matrix-vector multiplication with one output column.
