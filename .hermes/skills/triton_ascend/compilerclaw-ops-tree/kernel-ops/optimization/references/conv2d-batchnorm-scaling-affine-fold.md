# Conv2d + BatchNorm2d + scalar scale: affine fold and cache pattern

## Pattern

For models of the form:

```python
y = self.bn(self.conv(x)) * scale
```

avoid a full-output Triton scaling pass. Fold the scalar into BatchNorm affine parameters:

```python
x = self.conv(x)
weight = self.bn.weight * scale
bias = self.bn.bias * scale
return F.batch_norm(
    x,
    running_mean=self.bn.running_mean,
    running_var=self.bn.running_var,
    weight=weight,
    bias=bias,
    training=self.bn.training or not self.bn.track_running_stats,
    momentum=self.bn.momentum if self.bn.momentum is not None else 0.0,
    eps=self.bn.eps,
)
```

This preserves BatchNorm training semantics, including running-stat updates, while removing one launch and one full GM read/write pass over the output tensor.

## No-grad cache for tiny affine vectors

In inference/benchmark no-grad contexts, cache the scaled BN affine vectors behind data-pointer/version keys:

```python
if torch.is_grad_enabled():
    return self.bn.weight * s, self.bn.bias * s  # fresh autograd graph
key = (tensor_key(self.bn.weight), tensor_key(self.bn.bias), s, x.dtype, x.device)
if key != self._bn_affine_cache_key:
    self._bn_affine_cache_value = (self.bn.weight * s, self.bn.bias * s)
return self._bn_affine_cache_value
```

Do **not** reuse cached tensors when gradients are enabled; recompute to avoid stale autograd graphs.

## Eval-mode Conv+BN+scale fold

When `not self.bn.training` and running stats are tracked, fold Conv2d, BatchNorm, and scale into one Conv2d:

```python
inv_std = torch.rsqrt(running_var + eps)
g = gamma * scale * inv_std
b = beta * scale + (conv_bias - running_mean) * g
W_fused = W * g.view(-1, 1, 1, 1)
return F.conv2d(x, W_fused, b, stride=..., padding=..., dilation=..., groups=...)
```

Cache `(W_fused, b)` with tensor data-pointer/version keys for weights, bias, BN affine tensors, and running stats. Mention that `.data` mutation can bypass versioning and requires manual cache clearing.

## Profiling requirement

If the optimized path has no-grad/inference caches, benchmark inside `torch.no_grad()` so timing reflects the intended cached path and avoids autograd overhead:

```python
def timed():
    with torch.no_grad():
        return model(x)
```

## Cannsim note

A cannsim trace of the old scalar scale epilogue remains useful as evidence of the eliminated work. The production optimization is launch/GM-pass removal, so report the scale-kernel diagnostic trace separately from end-to-end remote hardware latency.
