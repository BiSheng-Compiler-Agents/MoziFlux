# Optimizations Applied

## 1. Transpose weight once to make B tile loads contiguous

Baseline reads `weight[N, K]` as a logical `[K, N]` tile with `stride_bn = K`, so columns in the `N` dimension are strided by the full K dimension:

```python
b_ptrs = b_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
```

The optimized host materializes `weight.T` as contiguous `[K, N]` before launching the kernel:

```python
b_kn = b.transpose(0, 1).contiguous()
...
b_ptrs = b_base + k_idxs[:, None] * stride_bk  # stride_bn == 1, stride_bk == N
```

This makes each B row contiguous along `N`, improving MTE coalescing. The transpose cost is small relative to the 1024×8192×8192 GEMM.

## 2. In-place Cube accumulation

Baseline creates a separate `tl.dot` result and then vector-adds it into `acc`:

```python
acc += tl.dot(a, b)
```

The optimized kernel accumulates directly in the Cube operation:

```python
acc = tl.dot(a, b, acc)
```

This removes the fp32 temporary tile and reduces vector load/store/add pressure around the GEMM accumulator.

## 3. Static `tl.range` K loop with `care_padding=False`

The baseline uses a dynamic `while k < K` loop and default padding checks:

```python
k = 0
while k < K:
    ...
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    acc += tl.dot(a, b)
    k += BLOCK_K
```

The optimized kernel uses the Ascend-friendly range form and disables padding checks for masked matmul loads:

```python
for k0 in tl.range(0, K, BLOCK_K_):
    a = tl.load(..., mask=..., other=0.0, care_padding=False)
    b = tl.load(..., mask=..., other=0.0, care_padding=False)
    acc = tl.dot(a, b, acc)
```

Masked zero inputs do not affect dot accumulation, so skipping padding checks is safe here.

## 4. Larger output tile for large fp16 GEMM

The optimized fp16 large-GEMM path uses `BLOCK_M=64, BLOCK_N=256, BLOCK_K=64` instead of the baseline large-shape `64×64×64` tile.

```python
BLOCK_M = 64
BLOCK_N = 256
BLOCK_K = 64
```

A larger `N` tile amortizes scalar/control overhead over twice as many output elements while remaining within the 192 KB UB budget due to in-place dot accumulation.

## 5. Bounded 1D AI-Core scheduling

The optimized host launches at most the physical AI-Core count and lets each program process multiple output tiles:

```python
num_aicore = driver.active.utils.get_device_properties(device)["num_aicore"]
grid = (min(total_tiles, num_aicore),)
```

Inside the kernel:

```python
for tile_id in range(pid, total_tiles, tl.num_programs(0)):
    pid_m = tile_id // NUM_BLOCKS_N
    pid_n = tile_id - pid_m * NUM_BLOCKS_N
```

This avoids large multidimensional launch grids and preserves full tile coverage.

## 6. Safe fallback for unsupported/small regimes

The custom Triton path is tuned for the source problem's large fp16 GEMM. Smaller tensors and fp32/bf16 inputs use the native PyTorch/ACL path:

```python
if a.dtype != torch.float16 or m < 256 or n < 1024 or k < 1024:
    return torch.relu(torch.matmul(a, b.transpose(0, 1)) + bias)
```

This preserves the original interface for all supported dtypes without introducing shape errors in regimes not targeted by the custom kernel.
