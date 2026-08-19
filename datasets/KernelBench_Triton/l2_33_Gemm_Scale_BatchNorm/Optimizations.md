# Optimizations Applied

## 1. Replaced dynamic `while` K-loop with `tl.range`

Baseline:
```python
k_iter = 0
while k_iter < K:
    ...
    acc += tl.dot(a, b)
    k_iter += BLOCK_K
    a_ptrs += BLOCK_K * stride_ak
```

Optimized:
```python
for k_start in tl.range(0, K, BLOCK_K):
    a = tl.load(... (k_start + offs_k) ...)
    b = tl.load(... (k_start + offs_k) ...)
    acc = tl.dot(a, b, acc)
```

Rationale: `tl.range` removes pointer-advance scalar loop overhead and lets Triton-Ascend schedule the K loop more statically. This reduced control-flow events in the sub-kernel trace from 12 FLOWCTRL ops to 6 for a tile with 4× more output elements.

## 2. Used in-place Cube accumulation: `tl.dot(a, b, acc)`

Baseline:
```python
acc += tl.dot(a, b)
```

Optimized:
```python
acc = tl.dot(a, b, acc)
```

Rationale: in-place accumulation avoids a separate fp32 dot temporary in UB and accumulates through Cube hardware directly. This is the main Ascend-specific matmul optimization for reducing vector-side load/store pressure around `tl.dot`.

## 3. Cached a contiguous `[K, N]` weight transpose

Baseline weight access used `nn.Linear.weight` as `[N, K]` and read B with a stride of `K` across columns:
```python
B = self.gemm.weight.contiguous()
b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn  # stride_bn = K
```

Optimized host path:
```python
self._cached_weight_kn = self.gemm.weight.transpose(0, 1).contiguous()
B = self._cached_weight_kn  # [K, N]
```

Rationale: the kernel consumes B as `[K, N]`, so `offs_n` is contiguous (`stride_bn=1`). This improves MTE2 coalescing for the B tile and amortizes the transpose across repeated forwards using the parameter data pointer/version as a cache key.

## 4. Increased GEMM tile size and switched to 1D grouped scheduling

Optimized kernel uses `BLOCK_M=128, BLOCK_N=128, BLOCK_K=32` and maps the logical 2D tile space to one 1D launch grid:
```python
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
GROUP_M: tl.constexpr = 4
...
pid_m = first_pid_m + (pid_in_group % group_size_m)
pid_n = pid_in_group // group_size_m
```

Rationale: the larger tile increases Cube work per program while staying within UB/L1 constraints for fp32 inputs and accumulators. Grouped 1D scheduling improves reuse of B tiles and avoids a multidimensional launch grid.

## 5. Added `dot_pad_only_k` compile hint

```python
import triton.language.extra.cann.extension as al
...
al.compile_hint(acc, "dot_pad_only_k")
```

Rationale: `BLOCK_M` and `BLOCK_N` are already 16-aligned, so only K padding is needed. The hint avoids unnecessary M/N padding work in the Cube path.
