# ConvTranspose ACL Dispatch and Cannsim-Limited Baseline Probes

Use this reference when optimizing standard transposed convolution operators (`ConvTranspose1d/2d/3d`) whose editable baseline implements direct convolution with nested scalar/vector Triton loops.

## Recognition pattern

- The operator is a mature ACL-covered primitive: `torch.nn.ConvTranspose*` / `torch.nn.functional.conv_transpose*`.
- The custom Triton baseline computes output points directly with loops over input channels and kernel taps, e.g. `for ic in tl.static_range(...): for k in tl.static_range(...): acc += w * x`.
- The kernel does not use `tl.dot`/Cube and is structurally unlikely to beat ACL for standard convolution semantics.
- Realistic cannsim probes may fail before trace generation because the unrolled direct-convolution body spills too many vector slots or crashes BiSheng. Shrinking `BLOCK_*`, `Cin`, and `K` can produce a diagnostic micro-probe, but it may no longer represent full per-output work.

## Preferred optimization

Preserve module construction and parameter ownership, but dispatch forward to ACL:

```python
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=False):
        super().__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias,
        )

    def forward(self, x):
        mod = self.conv1d_transpose
        return torch.nn.functional.conv_transpose1d(
            x, mod.weight, mod.bias,
            stride=mod.stride,
            padding=mod.padding,
            output_padding=mod.output_padding,
            groups=mod.groups,
            dilation=mod.dilation,
        )
```

Do not add new guards beyond the baseline/module contract. Preserve `output_padding` and `groups` arguments even when the benchmark defaults are `0` and `1`.

## Cannsim reporting when trace generation fails

Still invoke `cannsim_local_run` on the baseline custom kernel. If realistic probes fail to compile or emit no `trace_core0.json`, report the exact bounded attempts and why they failed instead of inventing a trace table:

| Path | Probe | cannsim result | Cycles / latency | Trace availability |
|---|---:|---|---:|---|
| Baseline Triton | realistic/reduced/micro probe | compile spill/crash or device task only | observed cycles if present | unavailable with reason |
| Optimized | ACL dispatch | custom Triton launch removed | 0 cycles | not applicable |

The performance decision should then be gated by `remote_verify` hardware correctness and latency, with comparison Triton providers pre-skipped as `inf` if launching them would be slow, toxic, or structurally irrelevant.
