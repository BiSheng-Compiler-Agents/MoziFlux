# Conv2d with standard reduction/activation epilogue: ACL dispatch pattern

Use this reference when the main convolution is already ACL-backed (or should be ACL-backed) and the custom Triton work is only a standard post-conv epilogue such as channelwise/global `min`/`max`/`mean` plus `tanh`/`sigmoid`/other mature activations.

## Recognition

- `ModelNew.forward()` runs `nn.Conv2d` / `F.conv2d`, then launches a custom Triton epilogue.
- The epilogue is a standard ACL-covered reduction or activation chain, not a fused math pattern requiring custom data reuse.
- The Triton epilogue uses strided channel-plane loads, unsupported/non-portable intrinsics (for example `tl.tanh` on triton-ascend), or has meaningful MTE/WAIT/PUSHQ overhead in a grid=1 probe.

## Optimization pattern

Preserve constructor/parameter semantics and route the epilogue through ACL/PyTorch:

```python
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew only supports Ascend NPU execution")
        y = F.conv2d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return torch.tanh(torch.tanh(torch.amin(y, dim=1, keepdim=True)))
```

Adapt the reduction dimension and activation chain to the source kernel. Do not add new shape guards beyond the baseline contract.

## Cannsim reporting

- Trace only the removed custom epilogue with a sub-kernel probe if the full post-conv tensor is too large.
- Keep the epilogue body structurally similar, but reduce channel/spatial constants as needed for stable simulation.
- Report optimized custom Triton cycles as `0` only when the custom Triton launch is completely removed.
- Hardware latency must come from `remote_verify`, not from cannsim.

## Profiling pattern

- Keep parser-visible provider columns for PyTorch/ACL, baseline1, baseline2, and optimized.
- If comparison Triton baselines are toxic (unsupported intrinsic, read-only base, grid/context risk), pre-skip them with neutral `SKIP`/`inf` cells.
- Gate `UNIT_TEST PASS` on optimized correctness against PyTorch/ACL across small, medium, and exact shapes.

## Example observation

For a Conv2d -> channelwise minimum -> tanh -> tanh epilogue, a grid=1 sub-kernel probe showed MTE/WAIT dominated custom epilogue work (wall_cycles=3439; MTE3=2477 busy cycles, MTE2=1640, VEC=1609, PUSHQ=953). Replacing the epilogue with ACL `torch.amin` + `torch.tanh` removed the custom Triton launch and matched PyTorch/ACL hardware latency within noise while preserving exact outputs.
