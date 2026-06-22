# Optimizations

## 1. Replaced row-streaming product loop with block-level `tl.cumprod`

Baseline pattern streamed 8 scalar rows per loop and multiplied four 1D accumulators:

```python
while m + (UNROLL - 1) < M:
    v0 = tl.load(ptr + (m + 0) * stride_m, mask=mask_k, other=1.0)
    ...
    v7 = tl.load(ptr + (m + 7) * stride_m, mask=mask_k, other=1.0)
    acc0 *= (v0 * v1)
    ...
```

Optimized pattern loads a `[BLOCK_M, BLOCK_K]` tile and uses vector scan to collapse the whole M block:

```python
vals = tl.load(
    x_ptr + b * stride_b + m_idxs[:, None] * stride_m + offs_k[None, :] * stride_k,
    mask=(m_idxs[:, None] < M) & mask_k[None, :], other=1.0,
).to(tl.float32)
scan = tl.cumprod(vals, axis=0)
last_rel = tl.minimum(BLOCK_M, M - m0) - 1
block_prod = tl.sum(tl.where(offs_m[:, None] == last_rel, scan, 0.0), axis=0)
acc *= block_prod
```

Rationale: target `M=256` drops from 32 row-streaming iterations to 4 block iterations with `BLOCK_M=64`, reducing control-flow and MTE events while preserving exact product semantics, including negative values.

## 2. Persistent 1D launch cap

```python
n_k_tiles = triton.cdiv(K, block_k)
total_tiles = B * n_k_tiles
n_programs = min(total_tiles, 65535)
_prod_dim1_block_cumprod_kernel[(n_programs,)](..., total_tiles, n_k_tiles, n_programs)
```

Rationale: keeps Ascend `coreDim` legal for large `B*K` while the kernel grid-stride loop covers every output tile.

## 3. Contiguous K tiling and masked neutral padding

```python
offs_k_base = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), 16), BLOCK_K)
tl.load(..., mask=(m_idxs[:, None] < M) & mask_k[None, :], other=1.0)
```

Rationale: contiguous K offsets enable burst-friendly MTE access; `other=1.0` is the neutral product value for masked M/K tails.
