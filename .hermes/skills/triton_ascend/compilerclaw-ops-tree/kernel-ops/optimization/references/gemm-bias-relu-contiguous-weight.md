# GEMM + Bias + ReLU: Contiguous Weight-Tile Pattern

Use this reference when optimizing `nn.Linear`/GEMM-style kernels whose model weight is stored as PyTorch `[N, K]` but the Triton kernel consumes it as logical `[K, N]`.

## Symptom

A baseline may compute correct GEMM with `tl.dot`, but B-tile loads are strided in the output-channel dimension:

```python
# weight is [N, K], logical B tile is [K, N]
b_ptrs = weight_ptr + k[:, None] * stride_bk + n[None, :] * stride_bn
# for contiguous [N, K], stride_bn == K, so N lanes are non-contiguous
```

This pattern limits DMA coalescing and increases vector/load-store pressure. It often appears in fused `Gemm + bias + activation` kernels that pass `nn.Linear.weight` directly.

## General Fix

Materialize `weight.T` once on the host before launching the Triton kernel, then pass the contiguous `[K, N]` buffer:

```python
b_kn = weight.transpose(0, 1).contiguous()
...
b = tl.load(
    b_ptr + k[:, None] * stride_bk + n[None, :] * stride_bn,
    mask=k_mask[:, None] & n_mask[None, :],
    other=0.0,
    care_padding=False,
)
# now stride_bn == 1 and stride_bk == N
```

For large target GEMMs, the transpose/copy cost is usually small relative to the GEMM. If weights are static across repeated inference calls, consider caching the transposed weight in the module; if weights may change, keep the per-call transpose for correctness.

## Companion Optimizations

- Use `acc = tl.dot(a, b, acc)` instead of `acc += tl.dot(a, b)` to avoid the fp32 temporary accumulator tile.
- Keep matmul loads native fp16/bf16 where possible; accumulate in fp32.
- Use `tl.range(0, K, BLOCK_K)` rather than a manually incremented `while` loop.
- `care_padding=False` is safe for masked GEMM loads because masked zero lanes contribute zero to `tl.dot`.
- Try larger `BLOCK_N` after making B contiguous; compare normalized cycles per output element if tile sizes differ.
- For full-shape launches, a 1D AI-core-bounded loop can avoid excessive multidimensional launch grids:

```python
total_tiles = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)
grid = (min(total_tiles, num_aicore),)

@triton.jit
def kernel(..., NUM_BLOCKS_N: tl.constexpr, TOTAL_TILES: tl.constexpr):
    pid = tl.program_id(0)
    for tile_id in range(pid, TOTAL_TILES, tl.num_programs(0)):
        pid_m = tile_id // NUM_BLOCKS_N
        pid_n = tile_id - pid_m * NUM_BLOCKS_N
        ...
```

## Cannsim Reporting Note

If changing `BLOCK_N` from e.g. 128 to 256, the optimized sub-kernel computes twice as many output elements. Report both absolute `wall_cycles` and normalized `cycles/output_element`; otherwise an optimized trace can look slower despite doing more work.
