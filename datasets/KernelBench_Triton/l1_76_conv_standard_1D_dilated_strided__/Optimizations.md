# Optimizations

## 1. Removed scalar/vector direct Triton convolution launch

Baseline dispatches each `(batch, output_position, output_channel_tile)` to a Triton program and performs the convolution as scalar input loads plus vector weight FMAs:

```python
for ic in tl.static_range(0, IC):
    for k in tl.static_range(0, K):
        x_val = tl.load(...)
        w_vec = tl.load(...)
        acc += w_vec * x_val
```

This is a standard Conv1d primitive already covered by PyTorch/ACL. The optimized host interface keeps the same `nn.Conv1d` parameter ownership but calls ACL directly:

```python
return F.conv1d(
    x,
    conv1d.weight,
    conv1d.bias,
    stride=conv1d.stride,
    padding=conv1d.padding,
    dilation=conv1d.dilation,
    groups=conv1d.groups,
)
```

Rationale: the custom Triton path does not use `tl.dot`/Cube and has an invalid full launch product at the target shape: `N * L_OUT * ceil(OC/64) = 64 * 174758 * 2 = 22,369,024` programs, far above Ascend's 65,535 cap. ACL Conv1d uses the vendor convolution engine and avoids the custom scalar/MTE overhead entirely.

## 2. Preserved ModelNew initialization and output contract

```python
self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
```

The optimized model preserves constructor arguments, parameter initialization order, dtype/device movement through `nn.Module.to()`, no-padding semantics, stride=3, and dilation=4 behavior.
