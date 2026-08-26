# GEMM hybrid ACL/Triton dispatch

Use this when optimizing fused GEMM-style operators (`Linear/GEMM + scalar/bias/activation`) on Ascend.

## Pattern

1. Benchmark PyTorch/ACL as a first-class provider, not just the custom Triton baseline.
2. Keep the custom Triton path for large regimes only when hardware proves it wins.
3. Route small/medium GEMM regimes to ACL when Triton launch/autotune overhead dominates:

```python
if M < M_THRESHOLD or K < K_THRESHOLD or N < N_THRESHOLD:
    return fused_epilogue(torch.nn.functional.linear(x, weight, bias))
else:
    return _triton_fused_gemm(...)
```

4. Preserve exact module semantics: same `nn.Linear` weights/bias, same scalar multiplier, same activation slope, same dtype/device checks.
5. If output tile count can exceed Ascend's 65,535 launch cap, keep a grid-capped persistent fallback for legality, but do not route normal benchmark shapes through that fallback.

## Pitfalls

Do not assume matmul micro-optimizations (`tl.dot(a,b,acc)`, `care_padding=False`, `dot_pad_only_k`, `tl.range`) are universally faster for every fused fp32 GEMM. Apply them only when cannsim plus hardware verify show a win. If target-tile cannsim is MTE/FIXP/Cube-wait dominated and hardware shows parity/regression, prefer dispatch selection over forcing source-level changes.

For GEMM followed by reduction over the output hidden dimension, algebraic fusion can change fp32 reduction order:

```python
# Original order: may be required for strict atol=1e-3 at very large K/H
out = ((x @ weight.T) / 2).sum(dim=1, keepdim=True) * scale

# Fused order: faster GEMV, but validate max_abs at target size
s_eff = weight.sum(dim=0) * (scale * 0.5)
out = x @ s_eff[:, None]
```

If fused GEMV passes small/medium shapes but fails the exact target by a small max-abs drift, use hybrid dispatch: fused GEMV for regimes that pass tolerance and exact PyTorch/ACL reduction order for the target/large regime. Do not claim the fused path is generally correct until every benchmark shape passes the same tolerance as the original expression.

When the post-GEMM chain contains a row reduction with `keepdim=True`, inspect later operations for singleton-dimension identities before writing kernels. For shape `(B, 1)`, operations such as `max(dim=1, keepdim=True)`, `avg_pool1d(kernel_size=1)`, and `logsumexp(dim=1, keepdim=True)` are identities. The safe optimization pattern is:

```python
# Exact large/target path preserves original fp32 order.
if I >= LARGE_I and O >= LARGE_O:
    return F.linear(x, weight, bias).sum(dim=1, keepdim=True)

# Small/medium path can cache effective weights if validated.
wsum = weight.sum(dim=0).contiguous()
bsum = bias.sum().reshape(())
return rowwise_dot(x, wsum) + bsum
```

Cache `wsum`/`bsum` by parameter storage/version so steady-state inference does not rescan the full weight matrix, but always benchmark against ACL and validate the exact target separately for fp32 reassociation drift.

For N=1 GEMV, a custom Triton `tl.dot` path may need padding to `BLOCK_N=16`; cannsim can show this fallback slower due to extra SCALARLDST/FLOWCTRL. Treat it as diagnostic unless hardware proves it beats ACL.

For GEMM followed by a row-wise large-hidden reduction such as `logsumexp` plus scalar activations, keep ACL `F.linear` as the production GEMM unless a fused Cube kernel is separately proven. Then benchmark the post-op dispatch threshold: a cleaned Triton online row fallback may win for small/medium hidden sizes, while ACL `torch.logsumexp` can win or match for large hidden sizes. Do not route all sizes to ACL just because the target/default shape benefits; add a shape threshold (for example `if output_N >= target_hidden: ACL else Triton fallback`) only after remote hardware confirms it. Gate direct row-grid Triton by `B <= 65535`; for larger batches use ACL or an explicitly persistent row-loop fallback rather than risking `coreDim` overflow.

## Reporting

In `performance_report.md`, separate:
- cannsim sub-kernel trace for the Triton body; and
- hardware table showing dispatch-level speedup from ACL/Triton routing.

This avoids claiming tile-level cannsim speedup when the actual win comes from host dispatch.
