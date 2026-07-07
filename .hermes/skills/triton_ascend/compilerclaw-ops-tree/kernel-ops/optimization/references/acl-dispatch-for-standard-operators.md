# ACL Dispatch for Standard Operators

Use this reference when a KernelBench baseline implements a standard library-covered operator (Conv2d/Conv3d/pooling/etc.) with a custom Triton kernel that is structurally unlikely to beat ACL.

## Recognition pattern

- The operator is a mature primitive already exposed by `torch.nn`/`torch.nn.functional` on Ascend, including Conv1d/Conv2d/Conv3d and pooling variants.
- The custom Triton kernel is a direct scalar/vector implementation, not a hardware-appropriate algorithm.
  - Example symptoms: convolution expressed as nested `C*K` or `C*K*K` loops, weak/no Cube use, grid product near/exceeding Ascend's 65,535 launch cap, large SCALAR/PUSHQ/MTE pressure.
- The public deliverable is `ModelNew`; preserving constructor, parameter ownership, seed/init order, and `forward()` semantics matters more than preserving the custom launch.

## Optimization pattern

```python
class ModelNew(nn.Module):
    def __init__(...):
        self.conv2d = nn.Conv2d(...)

    def forward(self, x):
        return torch.nn.functional.conv2d(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups,
        )
```

Adapt the call for the relevant primitive (`conv3d`, pooling, etc.). Do not add new runtime guards beyond the baseline contract.

## Cannsim reporting

- Still run `cannsim_local_run` on the baseline custom Triton path to prove the removed work and identify the bottleneck.
- If a realistic tile is too large for stable local cannsim, shrink to a reliable micro-probe of the same kernel body and clearly label it as a scale-limited bottleneck probe.
- For the optimized path, report custom Triton device-kernel cycles as `0` because the custom launch was removed; physical latency must come from `remote_verify`.

## Profiling notes

- Keep all required provider columns (`PyTorch / ACL`, baseline(s), optimized) and parser-visible TEST entries.
- If comparison Triton baselines are huge/risky and not needed to gate optimized correctness, pre-skip them with neutral `SKIP`/`INF` wording while requiring optimized PASS on every benchmark shape.
- Document that baseline latency is unavailable/pre-skipped rather than fabricating a speedup over the skipped provider.
