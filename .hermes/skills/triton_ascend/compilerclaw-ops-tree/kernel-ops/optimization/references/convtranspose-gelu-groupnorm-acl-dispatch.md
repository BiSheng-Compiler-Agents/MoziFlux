# ConvTranspose + GELU + GroupNorm ACL Dispatch

Use this reference when optimizing a model that already uses `nn.ConvTranspose2d` / ACL for the convolution, then applies a custom Triton fused epilogue for exact GELU followed by GroupNorm.

## Recognition pattern

- `forward()` calls `self.conv_transpose(x)` using `nn.ConvTranspose2d` or functional ACL/PyTorch convolution.
- A custom Triton kernel then computes exact GELU and GroupNorm statistics/affine over the output tensor.
- The Triton epilogue is two-pass or otherwise reloads the same group data: one pass for `sum`/`sum_sq`, another pass for normalize + affine.
- Cannsim sub-kernel traces show scalar/dispatch pressure such as PUSHQ, SCALARLDST, SCALAR, MTE waits, or repeated `tl.erf`/GELU work.
- Target output is huge enough that custom full-shape timing can be toxic, slow, or not worth comparing against mature ACL primitives.

## Preferred optimization

Preserve the constructor, parameter ownership, and exact semantics, but dispatch the post-convolution work to ACL/PyTorch primitives:

```python
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.groups = groups  # preserve interface if the baseline accepted it but did not use it

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects NPU inputs")
        y = self.conv_transpose(x)
        y = torch.nn.functional.gelu(y, approximate="none")
        return torch.nn.functional.group_norm(
            y,
            self.group_norm.num_groups,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.eps,
        )
```

Do not reinterpret unused constructor arguments such as `groups` if the editable baseline did not pass them into the convolution module; preserving the public interface is safer than changing semantics.

## Cannsim and reporting

- Run cannsim on a bounded sub-kernel of the original custom epilogue to document the bottleneck.
- The optimized path may have no custom Triton kernel to simulate; report optimized custom work as `0 cycles / ACL dispatch`, not as a fabricated trace.
- Useful table columns: wall cycles, estimated latency (`cycles * 0.4 ns`), bottleneck pipeline, and top scalar/PUSHQ/MTE instructions.

## Profiling pattern

- Keep all providers parser-visible in `profile_kernels.py`: `PyTorch / ACL`, `Baseline Triton1`, `Baseline Triton2`, `Optimized Triton`.
- Unit-test optimized correctness on all benchmark shapes, including the target.
- If target custom baselines are structurally unsuitable/toxic, pre-skip only those comparison providers at target with neutral `SKIP`/`inf` cells while still benchmarking small/medium comparison shapes when safe.
- Gate success on optimized correctness and hardware benchmark from `remote_verify`, not on cannsim alone.
