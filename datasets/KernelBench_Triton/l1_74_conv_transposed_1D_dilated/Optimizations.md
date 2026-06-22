# Optimizations

## 1. Replace direct Triton ConvTranspose1d with ACL dispatch

Baseline custom path:
```python
_conv_transpose1d_kernel[grid](x_c, weight_c, bias_c if bias_c is not None else y, y, ...)
```

Optimized path:
```python
return F.conv_transpose1d(
    x, mod.weight, mod.bias,
    stride=mod.stride, padding=mod.padding,
    output_padding=mod.output_padding,
    groups=mod.groups, dilation=mod.dilation,
)
```

Rationale: ConvTranspose1d is a mature ACL-covered primitive. The baseline uses vector scalar multiply-add loops over `Cin*K` for every `(Cout, time)` tile, with no `tl.dot`/Cube mapping and a target launch product `B * ceil(Lout/128) * ceil(Cout/64) = 32 * 1025 * 1 = 32800` blocks; ACL avoids the custom vector-loop convolution and preserves `nn.ConvTranspose1d` parameter semantics.

## 2. Preserve constructor and benchmark contract

```python
self.conv1d_transpose = nn.ConvTranspose1d(
    in_channels, out_channels, kernel_size,
    stride=stride, padding=padding, dilation=dilation, bias=bias,
)
```

Rationale: `ModelNew` keeps the same initialization order and parameter ownership as the original module, so benchmark/reference seeding remains comparable while the forward path delegates to the optimized vendor implementation.
