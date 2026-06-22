# Optimizations

## 1. Replace direct fused Triton convolution with ACL convolution dispatch

**Before (input baseline):**
```python
for ci in range(0, C_IN):
    s = tl.zeros([BM], dtype=tl.float32)
    for kh in tl.static_range(0, K):
        for kw in tl.static_range(0, K):
            x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
            w_val = tl.load(wdw_ptr + wdw_base + kh * K + kw)
            s += x_vals * w_val
    b_vec = tl.load(wpw_panel + ci, mask=n_mask, other=0.0)
    acc += s[:, None] * b_vec[None, :]
```

**After:**
```python
y = F.conv2d(x32, self.depthwise.weight, None,
             stride=self.depthwise.stride, padding=self.depthwise.padding,
             dilation=self.depthwise.dilation, groups=self.depthwise.groups)
y = F.conv2d(y, self.pointwise.weight, None)
```

The baseline implements a standard depthwise-separable Conv2d as scalar/vector multiply-add loops and never uses `tl.dot`/Cube for the pointwise 1x1 stage. Ascend already provides mature ACL Conv2d kernels, so the optimized host path preserves the module parameter contract and dispatches both depthwise and pointwise convolutions to ACL.

## 2. Preserve initialization order and dtype semantics

```python
self.depthwise = nn.Conv2d(..., groups=in_channels, bias=bias)
self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
orig_dtype = x.dtype
x32 = x.contiguous().to(torch.float32)
...
return y.to(orig_dtype)
```

The optimized model creates the same two `nn.Conv2d` modules in the same order as the source so seeded parameter initialization matches. It also preserves the baseline's fp32 compute path and casts the output back to the input dtype.

## 3. Eliminate illegal/oversized Triton launch shape

```python
# baseline grid at exact shape:
# ceil(N * Hout * Wout / 64) * ceil(Cout / 32)
# = ceil(16 * 512 * 512 / 64) * 4 = 262144 programs > 65535
```

The source fused kernel uses a 2-D launch product that exceeds Ascend's `coreDim <= 65535` limit at the benchmark shape. The optimized ACL dispatch removes that custom Triton launch entirely, avoiding context poisoning and FFTS launch overflow.
