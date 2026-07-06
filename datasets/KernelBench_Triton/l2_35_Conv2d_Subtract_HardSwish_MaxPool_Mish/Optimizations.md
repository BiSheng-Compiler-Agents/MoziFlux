# Optimizations

## 1. Replaced the hot default path with ACL-backed standard operators

The editable baseline already uses `nn.Conv2d` for the convolution, then launches a Triton epilogue for subtract + HardSwish + MaxPool + Mish. Cannsim and hardware profiling showed that this Triton epilogue is dominated by scalar/spill overhead for the pooling index decomposition and transcendental Mish sequence. The optimized `forward()` keeps the convolution and dispatches the remaining standard ops to PyTorch/ACL kernels:

```python
x = self.conv(x)
v = x - self.subtract_value
hswish = v * torch.clamp(v + 3.0, min=0.0, max=6.0) * (1.0 / 6.0)
pooled = self.pool(hswish)
return pooled * torch.tanh(torch.nn.functional.softplus(pooled))
```

Rationale: these are all standard NPU-supported ops; ACL avoids the custom Triton epilogue's scalar DIV/REM, SCALARLDST spills, and PUSHQ pressure while preserving exact semantics for arbitrary square pool sizes.

## 2. Kept Triton helper kernels for experimental/direct fused paths, but did not route the hot path through them

A direct vectorized 2x2 pooling Triton variant was implemented and simulated:

```python
k = tl.arange(0, 4)
kh = k // 2
kw = k - kh * 2
vals = tl.load(x_ptr + base[None, :] + kh[:, None] * W + kw[:, None],
               mask=mask[None, :], other=0.0).to(tl.float32) - subtract_value
hs = vals * (tl.minimum(tl.maximum(vals + 3.0, 0.0), 6.0) * (1.0 / 6.0))
mv = tl.max(hs, axis=0)
```

Rationale: this tested the maxpool episode pattern (`tl.max(axis=0)` over all pool positions). Cannsim rejected it for this activation-heavy epilogue: the vectorized 2-D tile increased SCALARLDST/PUSHQ pressure and was slower than the scalar-loop subkernel, so the production host path uses ACL instead.

## 3. Grid-cap-safe fallback shape handling

The retained Triton helper path includes conservative grid-cap handling (`<= 65535`) and a row-persistent overflow helper for very large `K=2` outputs. The production `forward()` is ACL-backed and therefore avoids Ascend FFTS grid-cap failures entirely for the benchmark and fallback `pool_k=3` dispatch paths.

## Correctness coverage

`profile_kernels.py --test` covers:
- `default_k2`: required benchmark shape `(128, 64, 128, 128)`, `pool_k=2`
- `small_k2`: irregular spatial shape `(4, 64, 65, 67)`, `pool_k=2`
- `fallback_k3`: non-default pool size, `pool_k=3`

Remote verification result: `UNIT_TEST PASS` for optimized output on all shapes.