# Singleton-Channel Softmax Constant-Output Elimination

Use this when a KernelBench-style operator reduces a channel dimension with `keepdim=True` and then applies `softmax(dim=channel_dim)`.

## Recognition pattern

```python
y = expensive_op(x)                   # e.g. ConvTranspose2d/3d
y = y.mean(dim=1, keepdim=True)       # channel dimension becomes 1
y = y + bias                          # optional broadcast add
y = torch.softmax(y, dim=1)           # singleton softmax
y = torch.tanh(y) * scaling_factor    # optional unary/scale tail
```

After the `keepdim=True` reduction, the softmax dimension has size 1, so `softmax(..., dim=1)` is exactly `1` for every element. Any values produced before the singleton softmax are mathematically discarded, including convolution output and bias.

## Preferred optimization

Preserve constructor/module fields for API and state-dict compatibility, but in `forward()` compute only the output shape and fill a constant tensor:

```python
out = torch.empty((N, 1, Do, Ho, Wo), device=x.device, dtype=x.dtype)
const_val = math.tanh(1.0) * float(self.scaling_factor)
_fill_const_kernel[(n_tiles,)](out, const_val, out.numel(), BLOCK_SIZE=8192)
return out
```

For ConvTranspose output shape, use the framework formula:

```python
Do = (Di - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
```

## Dispatch and profiling notes

- Use a UB-safe constant-fill tile, commonly `BLOCK_SIZE=8192` for fp32.
- Keep direct dispatch for `ceil(numel / BLOCK_SIZE) <= 65535`; add a separate persistent tile-loop path only for oversized valid outputs.
- If the editable input has a constructor positional quirk, the optimized constructor can accept both forms without modifying `base_*.py`.
- In `profile_kernels.py`, comparison providers that are constructor-incompatible or read-only/toxic should remain parser-visible and be skipped neutrally as `inf`; correctness gates on the optimized provider and the PyTorch/ACL reference.

## Cannsim interpretation

A constant-fill trace is usually fixed-overhead/store-bound (`SCALAR`, `MTE3`, `RVECST`). Compare cycles per element when changing tile size: a larger UB-safe tile can improve normalized throughput even if wall cycles stay nearly flat.
