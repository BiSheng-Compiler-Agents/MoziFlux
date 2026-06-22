# Optimizations Applied

## 1. Split no-index hot path from index-tracking path

The default workload uses `return_indices=False`, so the optimized file dispatches it to dedicated no-index kernels and keeps an index kernel only for the optional API path.

```python
if not self.return_indices:
    _maxpool1d_noindex_persistent_kernel[...]  # or direct no-index path
else:
    _maxpool1d_index_kernel[...]
```

Rationale: the baseline carries index-related control flow inside the same kernel. Specializing the hot path lets the compiler delete `chosen_pos` updates and only compute the pooled value.

## 2. Fast interior-tile path removes padding/bounds work

Most target tiles are interior tiles: with `L_in=65536`, `K=8`, `stride=1`, `padding=4`, and `dilation=3`, only boundary tiles need clamp/valid checks.

```python
block_start = pid_o_blk * BLOCK * STRIDE - PADDING
block_last = (pid_o_blk * BLOCK + BLOCK - 1) * STRIDE - PADDING + (K - 1) * DILATION
full_tile = (pid_o_blk * BLOCK + BLOCK <= L_out) & (block_start >= 0) & (block_last < L_in)

if full_tile:
    for k in tl.static_range(0, K):
        xk = tl.load(base_x + (starts + k * DILATION).to(tl.int64),
                     mask=mask_o, other=-float("inf")).to(tl.float32)
        y_max = tl.maximum(y_max, xk)
else:
    ...  # generic clamped boundary path
```

Rationale: the baseline performs `pos>=0`, `pos<L_in`, `min`, `max`, and `tl.where` for every window element. The fast path removes those scalar operations for interior tiles; cannsim wall cycles fell from `16222` to `4012` on the representative full-tile subkernel.

## 3. Grid-capped persistent dispatch for the large target shape

The target shape has `NC * cdiv(L_out, 128) = 64 * 192 * 512 = 6,291,456` logical tiles. The optimized no-index path caps the launch to physical vector cores and loops over tiles inside each program.

```python
if total_tiles > _MAX_PROGRAMS:
    n_programs = max(1, min(core_num, _MAX_PROGRAMS, total_tiles))
    _maxpool1d_noindex_persistent_kernel[(n_programs,)](...)
else:
    _maxpool1d_noindex_direct_kernel[(NC, n_o_blks)](...)
```

Rationale: this avoids oversized FFTS/coreDim-style launches while preserving a direct path for small inputs. The benchmark/unit-test script covers both the direct path and the persistent path.

## 4. `return_indices=True` correctness fallback

```python
if self.return_indices:
    return F.max_pool1d(x, self.kernel_size, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        ceil_mode=False, return_indices=True)
```

Rationale: the optional indices path is not the benchmark hot path, and the Triton int64-index kernel hit a backend compile failure on remote hardware. The host interface keeps the API correct by routing that optional path to PyTorch/ACL while the optimized Triton kernels handle the default no-index path.

## 5. FP32 reduction value and masked loads

```python
y_max = tl.full((BLOCK,), -float("inf"), tl.float32)
xk = tl.load(..., mask=valid, other=-float("inf")).to(tl.float32)
y_max = tl.maximum(y_max, xk)
```

Rationale: max reduction is computed in FP32 for stable comparison across fp16/bf16/fp32 inputs, and every load/store remains masked for Ascend out-of-bounds safety.
