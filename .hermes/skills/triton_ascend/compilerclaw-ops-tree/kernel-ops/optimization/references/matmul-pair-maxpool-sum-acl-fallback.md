# Matmul + adjacent-pair MaxPool + row sum: ACL production with Cube fallback

Use this when a KernelBench operator is `Linear(x)` followed by `MaxPool1d(kernel_size=2, stride=2)`, `sum(dim=1)`, and a scalar scale at a large aligned GEMM shape.

## Pattern

1. **Production path:** dispatch the standard chain to ACL/PyTorch operators:

```python
z = F.linear(x, weight, bias)
pooled = F.max_pool1d(z.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
return pooled.sum(dim=1) * scale
```

This preserves ACL reduction/order semantics and avoids a custom Vector-Core matmul body for very large aligned GEMM.

2. **Triton fallback / cannsim path:** keep a real custom kernel that computes adjacent output pairs with Cube:

```python
acc0 = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
acc1 = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
for k0 in tl.range(0, K, BLOCK_K):
    acc0 = tl.dot(x_tile, w_even, acc0)
    acc1 = tl.dot(x_tile, w_odd, acc1)
pair_max = tl.maximum(acc0 + b_even[None, :], acc1 + b_odd[None, :])
partial = tl.sum(pair_max, axis=1)
```

Then reduce the per-row partials in a second lightweight kernel. This avoids atomics and demonstrates the custom path uses Cube instead of the baseline `tl.sum(w * x)` vectorized matmul.

## Cannsim reporting

- If the baseline vector matmul sub-kernel times out, shrink the microprobe aggressively (one row, one pool pair, very small `IN`) and state that normalized cycles/MAC are the fair comparison.
- Do not compare absolute wall cycles across probes with different represented work. Report `cycles/MAC` or `ns/MAC`.
- Scope claims: cannsim covers the Triton fallback body; production hardware latency must come from `remote_verify`.

## Profiling requirements

- Include ACL production (`Optimized Triton`) and forced Triton fallback (`Optimized TritonFallback`) in tests.
- If `base_*.py` is read-only or unavailable, keep the Baseline Triton2 column and emit parser-visible TEST SKIP/UNAVAILABLE lines.
