# Optimizations

## 1. Replace fragile custom reduction with ACL/PyTorch reduction path

**Before** the editable Triton baseline performed the post-convolution min/sum/GELU/bias chain inside one nested-loop Triton kernel:

```python
while (h + 3) < H:
    ...
    while c_start < C:
        v0 = tl.load(..., cache_modifier=".cg")
        min0 = tl.minimum(min0, tl.min(v0, axis=0))
...
tl.store(out_ptrs, gelu_vals[None, :] + bias_vals[:, None], ...)
```

**After** the optimized host keeps the standard ConvTranspose/min/sum/GELU operations on ACL/PyTorch, which avoids the BiSheng nested-reduction compile abort seen in cannsim and uses vendor kernels for the expensive work:

```python
x = self.conv_transpose(x).contiguous()
reduced = x.min(dim=1, keepdim=True).values.sum(dim=2, keepdim=True)
gelu_vals = F.gelu(reduced)
```

Rationale: this operator is dominated by a standard ConvTranspose2d plus reductions. The baseline custom post-kernel could not be compiled by the Ascend backend in a bounded cannsim probe (`Not all operands are collapsed` / `Collapser.cpp` assertion), so the optimized implementation avoids that fragile nested reduction path.

## 2. Keep only a small Triton epilogue for broadcast bias add

The remaining custom Triton work is the final `[N, 1, 1, W] + [B, 1, 1] -> [N, B, 1, W]` epilogue:

```python
vals = tl.load(tmp_ptr + pid_n * W + w_offsets, mask=mask_w, other=0.0).to(tl.float32)
bias = tl.load(bias_ptr + b_offsets * sbc, mask=mask_b, other=0.0).to(tl.float32)
out = vals[None, :] + bias[:, None]
tl.store(out_ptrs, out, mask=mask_b[:, None] & mask_w[None, :])
```

Rationale: the epilogue has simple contiguous vector loads/stores and compiles cleanly for cannsim. It also covers both `bias_shape=(1,1,1)` and multi-channel bias shapes without the baseline's OOB risk when bias has one channel.

## 3. Preserve safe fallback for CPU/non-fp32 dispatch

```python
if gelu_vals.dtype != torch.float32 or gelu_vals.device.type != "npu":
    return gelu_vals + self.bias
```

Rationale: the Triton epilogue is specialized for the KernelBench fp32 NPU path. The fallback preserves correctness for local CPU tests and non-fp32 inputs instead of adding unsupported dtype assumptions.
