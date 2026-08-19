# ConvTranspose + Min/Sum/GELU/Add Hybrid Dispatch

Use this when a KernelBench-style operator does `ConvTranspose2d -> min(dim=channel) -> sum(dim=height) -> GELU -> bias add`, and the editable Triton baseline implements the post-convolution reduction with nested `while` loops and `tl.min` over channel tiles.

## Recognition pattern

- The convolution itself is already a standard `nn.ConvTranspose2d` ACL/PyTorch module.
- The custom Triton post-kernel performs a 2D tile reduction like `[BLOCK_C, BLOCK_W] -> [BLOCK_W]`, often inside nested `while h < H` / `while c_start < C` loops.
- The baseline may include `cache_modifier=".cg"` and `tl.min(v, axis=0)` reductions for every height row.
- A bounded cannsim compile probe can fail before trace generation with BiSheng/HFusion errors such as `Not all operands are collapsed` or a `Collapser.cpp` assertion. Treat this as a compiler-aborting provider, not as a measurable baseline trace.

## Preferred optimization

Delegate the expensive standard pieces to ACL/PyTorch and keep only a tiny Triton epilogue if it is simple enough to compile and verify:

```python
x = self.conv_transpose(x).contiguous()
reduced = x.min(dim=1, keepdim=True).values.sum(dim=2, keepdim=True)
gel = torch.nn.functional.gelu(reduced)

# Optional NPU fp32 Triton epilogue for [N,1,1,W] + [B,1,1] -> [N,B,1,W]
if gel.dtype != torch.float32 or gel.device.type != "npu":
    return gel + self.bias
return triton_bias_expand(gel.reshape(N, W).contiguous(), self.bias.contiguous())
```

The epilogue kernel should use masked contiguous loads/stores only:

```python
vals = tl.load(tmp + pid_n * W + w_offsets, mask=mask_w, other=0.0).to(tl.float32)
bias = tl.load(bias + b_offsets * sbc, mask=mask_b, other=0.0).to(tl.float32)
tl.store(out_ptrs, vals[None, :] + bias[:, None], mask=mask_b[:, None] & mask_w[None, :])
```

## Cannsim and profiling guidance

- Still run `cannsim_local_run` on the baseline probe. If BiSheng aborts, document the exact compile failure and mark the baseline trace as unavailable; do not invent cycles.
- Run `cannsim_local_run` on the tiny optimized epilogue if present. A scalar-heavy trace for the epilogue is acceptable because it is not the dominant full operator cost.
- In `profile_kernels.py`, keep `Baseline Triton1` and `Baseline Triton2` columns visible but pre-skip compiler-aborting providers as `inf` with neutral `INFO` wording to avoid poisoning the NPU context.
- Hardware may show the pure ACL path is slightly faster than ACL+Triton epilogue for `bias_shape=(1,1,1)`. If target-only latency matters, route the singleton-bias case to pure ACL/PyTorch add and reserve the epilogue for multi-bias expansion shapes.

## Review points

- Confirm the optimized code preserves arbitrary `bias_shape[0]` accepted by the constructor; do not hardcode output channels.
- The CPU/non-fp32 fallback should return the PyTorch expression directly.
- Static review should note that baseline trace absence is due to compiler abort, not correctness success or failure.
