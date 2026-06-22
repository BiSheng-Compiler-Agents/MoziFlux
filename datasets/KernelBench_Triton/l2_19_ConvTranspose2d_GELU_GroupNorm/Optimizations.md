# Optimizations

## 1. Replace two-pass custom Triton GELU+GroupNorm with ACL dispatch

Baseline code computes GELU values once for the GroupNorm statistics and then reloads the same output group for a second normalization/affine pass:

```python
while idx < group_elems:
    x = tl.load(x_group_ptr + i, mask=mask, other=0.0).to(tl.float32)
    z = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    acc1 += z
    acc2 += z * z
...
while ch < cpg:
    x = tl.load(x_group_ptr + ch_base + idx_hw, mask=mask_hw, other=0.0).to(tl.float32)
    z = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    tl.store(y_group_ptr + ch_base + idx_hw, y, mask=mask_hw)
```

The optimized host keeps the same ConvTranspose2d and GroupNorm parameter ownership but routes the mature operations to ACL/PyTorch:

```python
y = self.conv_transpose(x)
y = F.gelu(y, approximate="none")
return F.group_norm(y, self.group_norm.num_groups,
                    self.group_norm.weight, self.group_norm.bias,
                    self.group_norm.eps)
```

Rationale: ConvTranspose2d, GELU, and GroupNorm are standard ACL-covered primitives; the custom Triton epilogue is scalar/PUSHQ-heavy and reloads the same group data. The change preserves exact GELU (`approximate="none"`) and GroupNorm affine semantics while eliminating the custom launch bottleneck.

## 2. Preserve constructor and benchmark input contract

```python
def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
    self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
    self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
    self.groups = groups
```

Rationale: the public `ModelNew` constructor and `get_inputs()` / `get_init_inputs()` remain compatible with the editable baseline. `groups` was not used by the original ConvTranspose2d construction, so it is stored for interface preservation but not newly applied.
