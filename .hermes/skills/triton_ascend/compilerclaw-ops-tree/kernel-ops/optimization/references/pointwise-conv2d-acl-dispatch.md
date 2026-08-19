# Pointwise Conv2d ACL dispatch for custom GEMM-style baselines

Use this reference for KernelBench 1x1 / pointwise Conv2d variants where the editable baseline implements `Conv2d(kernel_size=1)` as a custom Triton GEMM-style kernel over `M=B*H*W` by `N=C_out`.

## Recognition

- `ModelNew` owns an `nn.Conv2d(..., kernel_size=1, stride=1, padding=0)` module.
- `forward()` flattens spatial positions into `M = B * H * W`, transposes weights with `weight.view(C_out, C_in).t().contiguous()`, and launches a custom `tl.dot` kernel over `(M, C_out)` tiles.
- Autotune configs may create an Ascend launch-grid edge case at large spatial shapes. Example: `ceil(B*H*W / 512) * ceil(C_out / 64)` can reach/exceed the 65,535 `coreDim` cap at target-scale inputs.
- cannsim full-shape runs are impractical; use a small 1-tile micro-probe to document the custom launch bottleneck if tracing is needed.

## Preferred optimization

Preserve the baseline constructor and parameter ownership, but dispatch the mature ACL Conv2d primitive:

```python
class ModelNew(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, bias=False):
        super().__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                stride=1, padding=0, bias=bias)

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects Ascend NPU tensors.")
        c = self.conv1d
        weight = c.weight if c.weight.dtype == x.dtype else c.weight.to(dtype=x.dtype)
        bias = None if c.bias is None else (c.bias if c.bias.dtype == x.dtype else c.bias.to(dtype=x.dtype))
        return torch.nn.functional.conv2d(
            x, weight, bias,
            stride=c.stride, padding=c.padding, dilation=c.dilation, groups=c.groups,
        )
```

Do not add new shape guards. The optimization is host dispatch: custom Triton kernel time becomes `0` because no custom kernel is launched; final latency must come from `remote_verify` hardware timing.

## Profiling pattern

- Keep all provider columns parser-visible: `PyTorch / ACL`, `Baseline Triton1`, `Baseline Triton2` if present, and `Optimized Triton`.
- Pre-skip risky comparison Triton baselines with neutral `SKIP` / `inf` wording when their exact custom launch can poison the NPU context or hit `coreDim` limits.
- Gate `UNIT_TEST PASS` on optimized correctness against PyTorch / ACL for representative and exact shapes.
- For exact multi-GB shapes where optimized and reference are both ACL dispatch, a shape/dtype check can be used to avoid duplicate expensive value comparison, but document this limitation in `performance_report.md`.
