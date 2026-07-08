# Constant singleton-softmax fill pattern

Use this when a KernelBench-style operator ends with `mean(..., keepdim=True)` or another reduction that produces shape `(B, 1)` followed by `softmax(dim=1)`. For finite intermediates, softmax over a size-1 dimension is exactly `1`, so preceding GEMM/dropout/activation work is dead for the final output.

## Optimization pattern

Preserve the original constructor/host interface, but avoid allocating unused heavy modules in the optimized model when their values cannot affect the result:

```python
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p

    def forward(self, x):
        out = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
        fill_ones_dispatch(out)
        return out
```

Use a small direct fill for default/small shapes and a larger direct fill for large batches. Keep a separate persistent fallback only when `cdiv(n_elements, BLOCK_SIZE) > 65535`.

```python
block = 128 if n_elements <= 128 else 1024
n_tiles = triton.cdiv(n_elements, block)
if n_tiles > _MAX_PROGRAMS:
    _fill_ones_persistent[(_MAX_PROGRAMS,)](out, n_elements, _MAX_PROGRAMS, BLOCK_SIZE=block)
else:
    _fill_ones_direct[(n_tiles,)](out, n_elements, BLOCK_SIZE=block)
```

## Profiling requirements

- Unit-test every dispatch path, including persistent fallback. Force persistent by temporarily lowering `_MAX_PROGRAMS`; do not allocate enormous tensors just to exceed the grid cap.
- If the active sandbox says `base_*.py` is `DO NOT read`, keep the Baseline Triton2 column and emit parser-visible `TEST Baseline Triton2 <label>: SKIP_UNAVAILABLE ... max_abs=inf` without importing it.
- In cannsim, one-program fill probes may report identical chip cycles for small and large blocks because fixed prologue/launch overhead dominates. Report normalized throughput (`elements / cycles`) in addition to absolute cycles.
- Use `num_stages=2` on Ascend and mask every store.

## Review caveat

This pattern assumes the intermediate value is finite. It matches standard KernelBench random finite inputs and existing analytic shortcuts; if NaN propagation is part of the operator contract, do not replace the chain with unconditional ones.
