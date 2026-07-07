# Direct convolution ACL dispatch and cannsim micro-probes

Use this reference for KernelBench Conv1d/Conv2d/ConvTranspose variants where the editable baseline implements convolution as scalar/vector direct loops instead of a hardware-appropriate Cube/ACL path.

## Recognition

- Standard convolution primitive already exists in `torch.nn` / `torch.nn.functional` on Ascend.
- Triton baseline loops over `IC*K` / `C*K*K` and accumulates `acc += w_vec * x_val` without `tl.dot`.
- Full-grid launch product can exceed Ascend's `coreDim <= 65535` limit, especially when the grid maps `(batch * L_OUT, output_channel_tiles)`.
- Exact compile-time loop constants may be too large for cannsim compilation; symptoms include VF stack spill such as `total stack object size ... exceeded vf stack size`.

## Optimization pattern

Prefer preserving the module parameter contract and dispatching to ACL:

```python
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, dilation=dilation, bias=bias)

    def forward(self, x):
        c = self.conv1d
        return torch.nn.functional.conv1d(
            x, c.weight, c.bias,
            stride=c.stride, padding=c.padding,
            dilation=c.dilation, groups=c.groups,
        )
```

Do not add new runtime guards beyond the baseline/module contract. Preserve constructor arguments, initialization order, and parameter ownership.

## Cannsim trace pattern

Still trace the removed Triton path, but use a reliable micro-probe when the exact body is too large:

1. Keep the same kernel body and nested-loop structure.
2. Keep output-channel tile size if possible (`BLOCK_OC=64` for vector-tile behavior).
3. Reduce compile-time loop constants only as needed to compile and finish, e.g. `IC=1, K=3` for Conv1d.
4. Launch `grid=(1,1,1)` and make the host validate non-zero output.
5. Report the exact compile limitation honestly in `performance_report.md` and label the trace as a scale-limited micro-probe.

Optimized custom-kernel cannsim cycles are reported as `0` only when the optimized path removes the custom Triton launch entirely and uses ACL. Physical latency must still come from `remote_verify`.

## Profiling pattern

Keep parser-visible provider columns:

- `PyTorch / ACL`
- `Baseline Triton1`
- `Baseline Triton2` if `base_*.py` exists
- `Optimized Triton`

If direct Triton baselines are unsafe or exceed grid limits, pre-skip them with neutral `SKIP` / `inf` wording. Gate `UNIT_TEST PASS` on optimized correctness over representative and exact shapes; for very large exact shapes where optimized and reference are identical ACL dispatch, a shape check plus same-dispatch statement can avoid a duplicate multi-GB reference allocation, but document this in the report.