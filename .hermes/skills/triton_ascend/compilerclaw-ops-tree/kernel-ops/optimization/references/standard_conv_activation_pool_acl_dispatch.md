# Standard Conv + Activation + Pool Chains: prefer ACL dispatch after tracing

## Pattern

For pipelines like `Conv2d -> subtract/bias -> HardSwish/ReLU/GELU -> MaxPool/AvgPool -> Mish/tanh/softplus`, do not assume a custom fused Triton epilogue is faster than PyTorch/ACL standard operators. The convolution is already ACL-backed in many KernelBench L2 baselines; the post-conv Triton epilogue can become scalar/spill-bound due to NCHW index decomposition, pool-window address math, and activation transcendental sequences.

## Diagnostic workflow

1. Keep `nn.Conv2d` / ACL for convolution unless the task explicitly requires replacing it.
2. Build a cannsim subkernel for the custom epilogue only.
3. If trace shows dominant `SCALARLDST`, `SCALAR`, or `PUSHQ` with low useful `RVECEX`, test ACL standard-op dispatch in the host path before spending time on more fusion.
4. If trying a Triton pool optimization, test it with cannsim before adopting. Loading all pool positions as a 2-D tile and using `tl.max(axis=0)` can improve simple pooling, but for activation-heavy conv epilogues it may increase spills and PUSHQ pressure.
5. For activation + AvgPool epilogues (`subtract -> tanh -> AvgPool -> subtract`, etc.), if a normal sub-kernel (`BLOCK_HW` 256-1024) keeps timing out or hits cannsim unsafe early-exit, shrink the diagnostic host aggressively (e.g. `BLOCK_HW=16`, one NC tile, `H=W=4`, `outH=outW=2`, `K=2`). This preserves scalar/indexing bottleneck evidence while keeping trace generation practical.
6. Make `profile_kernels.py` include all paths: required default pool size, irregular spatial shapes, and non-default pool sizes that exercise fallbacks.

## Example host replacement

```python
x = self.conv(x)
v = x - self.subtract_value
hswish = v * torch.clamp(v + 3.0, min=0.0, max=6.0) * (1.0 / 6.0)
pooled = self.pool(hswish)
return pooled * torch.tanh(torch.nn.functional.softplus(pooled))
```

## Evidence shape

A representative Conv2d + HardSwish + MaxPool + Mish case had a scalar/spill-bound custom Triton epilogue. A vectorized 2x2 pooling Triton candidate using `tl.max(axis=0)` regressed in cannsim, while ACL standard-op dispatch passed remote correctness and delivered large speedup versus the editable Triton baseline. The reusable lesson is the decision rule, not the particular latency numbers.
