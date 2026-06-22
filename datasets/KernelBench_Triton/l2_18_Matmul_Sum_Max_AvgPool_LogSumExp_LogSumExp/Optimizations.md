# Optimizations

## 1. Algebraic reduction of singleton tail ops

The operation chain reduces to the row sum of the linear output. After `sum(dim=1, keepdim=True)`, the tensor has one column, so `max`, `avg_pool1d(kernel=1)`, and both `logsumexp(dim=1, keepdim=True)` are identity operations.

```python
return F.linear(x, self.linear.weight, self.linear.bias).sum(dim=1, keepdim=True)
```

This removes the custom kernel path that recomputed a full effective weight vector from `W` inside every forward pass.

## 2. Cached effective weight for small/medium regimes

For smaller regimes where reassociation stays within tolerance, the model caches `weight.sum(dim=0)` and `bias.sum()` until the parameter storage/version changes.

```python
self._cached_wsum = self.linear.weight.sum(dim=0).contiguous()
self._cached_bsum = self.linear.bias.sum().reshape(())
```

The cached path avoids rereading the full `(out_features, in_features)` weight matrix in every inference call.

## 3. UB-sized rowwise fallback kernel

The fallback Triton kernel processes 16 rows by 256 features and loops over rows persistently when needed.

```python
for k0 in tl.range(0, I, BLOCK_K):
    x = tl.load(..., care_padding=False).to(tl.float32)
    w = tl.load(..., care_padding=False).to(tl.float32)
    acc += tl.sum(x * w[None, :], axis=1)
```

`care_padding=False` is safe because masked feature tails contribute zero to a linear dot product. The launch is capped by `min(n_tiles, 65535)` and the kernel loops over logical row tiles.

## 4. Exact ACL dispatch for the 8192×8192 target

Large fp32 GEMM+reduction can exceed `1e-3` tolerance if the reduction is algebraically reassociated into `x @ sum(W)`. The optimized model therefore routes the target regime to ACL `F.linear(...).sum(...)`, preserving the original reduction order while still avoiding the baseline's slow in-kernel scan of `W`.
