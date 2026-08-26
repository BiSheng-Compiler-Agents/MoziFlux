# Depthwise Conv2d ACL Dispatch Notes

Use this as a concrete reference for depthwise 2D convolution KernelBench-style operators when the editable baseline implements grouped Conv2d as a custom scalar/vector Triton kernel.

## Recognition

- `nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, ...)` with square input/kernel.
- Triton kernel maps programs over `(N*C, H_OUT, ceil(W_OUT/BLOCK_W))` and computes each `K x K` stencil with scalar/vector `tl.load` + multiply-add loops.
- Target shape grid product can exceed Ascend's launch limit even when each axis is individually reasonable.

Grid product guard example:
```python
grid_product = N * C * H_OUT * triton.cdiv(W_OUT, BLOCK_W)
if grid_product > 65535:
    # Pre-skip comparison Triton baseline in profile_kernels.py.
    return float("inf")
```

For the canonical `N=16, C=64, H=W=512, K=3, stride=1, padding=0, BLOCK_W=256` case:
`H_OUT=W_OUT=510`, so `16*64*510*ceil(510/256) = 1,044,480 > 65,535`.

## Optimization

Preserve constructor/parameter semantics and dispatch to ACL grouped Conv2d:
```python
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

Do not force an extra `.contiguous()` copy in the optimized host path unless the baseline contract requires it; ACL handles normal PyTorch layouts and the benchmark inputs are typically contiguous already.

## Cannsim reporting pattern

Run cannsim only on the custom Triton baseline sub-kernel. A useful micro-probe is one output row with `N=1`, `C=1`, `H=K`, `W=BLOCK_W+K-1`, `H_OUT=1`, `W_OUT=BLOCK_W`, `grid=(1,1,1)`. Report optimized custom-kernel cycles as `0 / N.A.` because no custom Triton device kernel remains; hardware latency comes from `remote_verify`.

Typical trace interpretation: SCALARLDST/SCALAR dominance and no Cube use confirms the custom kernel is structurally the wrong path for a mature convolution primitive.

## Profiling pattern

- Keep columns parser-visible: `PyTorch / ACL`, `Baseline Triton1`, `Baseline Triton2`, `Optimized Triton`.
- For full target shapes where custom baselines exceed grid guard, print neutral `SKIP grid_guard` TEST lines and return `inf` in benchmarks.
- Gate `UNIT_TEST PASS` on optimized correctness across all benchmark shapes; do not count read-only/comparison baseline guard skips as optimized failures.
