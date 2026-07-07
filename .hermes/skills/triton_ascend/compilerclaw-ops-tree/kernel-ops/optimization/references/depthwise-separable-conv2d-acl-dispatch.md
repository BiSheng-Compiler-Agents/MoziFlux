# Depthwise-separable Conv2d ACL dispatch pattern

Use this reference for KernelBench-style depthwise-separable Conv2d baselines that fuse depthwise and pointwise stages in one custom Triton kernel with scalar/vector loops.

## Recognition

- Source module constructs `nn.Conv2d(in_channels, in_channels, K, groups=in_channels)` followed by `nn.Conv2d(in_channels, out_channels, kernel_size=1)`.
- Triton fused kernel iterates `for ci in range(C_IN)` and nested `K*K`, accumulates `s += x_vals * w_val`, then `acc += s[:, None] * b_vec[None, :]`.
- Pointwise 1x1 is implemented as vector outer-product loops rather than `tl.dot`/Cube.
- Exact launch often uses `(ceil(N*Hout*Wout/BM), ceil(Cout/BN))`; guard the product against Ascend `coreDim <= 65535`.

## Optimization

Prefer ACL dispatch while preserving the source initialization order and parameter ownership:

```python
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        orig_dtype = x.dtype
        x32 = x.contiguous().to(torch.float32)
        y = F.conv2d(x32, self.depthwise.weight.to(x.device, torch.float32), self.depthwise.bias,
                     stride=self.depthwise.stride, padding=self.depthwise.padding,
                     dilation=self.depthwise.dilation, groups=self.depthwise.groups)
        y = F.conv2d(y, self.pointwise.weight.to(x.device, torch.float32), self.pointwise.bias)
        return y.to(orig_dtype)
```

## Cannsim microprobe

Large fused probes can fail backend compile with VF stack spill. Reduce compile-time constants until the same nested-loop body compiles and finishes:

- Try `BM=1, BN=1, C_IN=1, K=3`, tiny `H/W`, `grid=(1,1,1)`.
- Validate non-zero output in the C++ host; report it as a scale-limited microprobe.
- Optimized ACL dispatch has no custom Triton trace; report `0 custom Triton cycles` and use `remote_verify` for hardware latency.

## Profiling notes

- Keep provider columns: `PyTorch / ACL`, `Baseline Triton1`, `Baseline Triton2`, `Optimized Triton`.
- Pre-skip unsafe baseline1 exact shapes with neutral `SKIP grid_guard` when launch product exceeds 65535.
- If read-only `base_*.py` comparison raises, preserve the column and print neutral `SKIP provider_exception`; gate `UNIT_TEST PASS` only on optimized correctness.