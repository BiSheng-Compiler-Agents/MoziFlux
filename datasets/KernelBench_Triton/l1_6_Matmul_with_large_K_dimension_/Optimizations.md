# Optimizations

## 1. Split-K partial accumulation for large K

```python
if K >= _SPLIT_THRESHOLD_K:
    split_k = min(16, triton.cdiv(K, _BLOCK_K))
    partial = torch.empty((split_k, M, N), device=A.device, dtype=A.dtype)
    _matmul_splitk_partial_kernel[(grid_m, grid_n, split_k)](...)
    _splitk_reduce_kernel[(triton.cdiv(M * N, _REDUCE_BLOCK),)](...)
```

Rationale: the benchmark has only four output tiles but 524,288 K elements, so a standard 2-D grid under-utilizes AI cores. Split-K creates parallel K partitions and then reduces partial matrices; a two-kernel partial/reduce design was used instead of `tl.atomic_add` because fp32 atomics produced incorrect results on hardware for this shape.

## 2. In-place Cube accumulation

```python
acc = tl.dot(a, b, acc)
```

Rationale: in-place accumulation avoids the extra fp32 temporary implied by `acc += tl.dot(a, b)` and keeps accumulation in the Cube path.

## 3. `tl.range` K loop with hoisted masks

```python
row_mask = offs_m[:, None] < M
col_mask = offs_n[None, :] < N
for k_tile in tl.range(0, num_k_tiles):
    ...
```

Rationale: `tl.range` and precomputed row/column masks reduce scalar loop and comparison overhead in the K loop.

## 4. Ascend load/codegen hints

```python
a = tl.load(a_ptrs, mask=..., other=0.0, care_padding=False)
b = tl.load(b_ptrs, mask=..., other=0.0, care_padding=False)
al.compile_hint(a, "dot_pad_only_k")
al.compile_hint(b, "dot_pad_only_k")
```

Rationale: matmul padding contributes zero to the dot product, so `care_padding=False` is safe. `dot_pad_only_k` limits compiler padding to the K dimension.

## 5. Direct-path fallback

```python
if K >= _SPLIT_THRESHOLD_K:
    ...  # split-K path
else:
    _matmul_direct_kernel[(grid_m, grid_n)](...)
```

Rationale: split-K has extra launch and reduction overhead; small and non-power-of-two direct cases stay on the single-kernel path.
