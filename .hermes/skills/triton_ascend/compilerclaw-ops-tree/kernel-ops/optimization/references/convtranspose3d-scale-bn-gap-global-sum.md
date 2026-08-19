# ConvTranspose3d + scale + BatchNorm3d + GlobalAvgPool algebraic global-sum shortcut

Use this reference for KernelBench-style models with `nn.ConvTranspose3d`, scalar multiply, `nn.BatchNorm3d`, then `AdaptiveAvgPool3d((1,1,1))` / global average pooling.

## Recognition pattern

- Production path is eval mode BatchNorm with running stats (`not bn.training` and `track_running_stats=True`).
- `ConvTranspose3d` uses the simple geometry where every input element contributes the full kernel volume: `stride=(1,1,1)`, `padding=(0,0,0)`, `output_padding=(0,0,0)`, `dilation=(1,1,1)`, `groups=1`.
- Output is only the global average over spatial dimensions after scale and BatchNorm.
- Baseline materializes the full 5D transposed-convolution output, then BatchNorm/pool over all output elements.

## Preferred optimization

Avoid materializing the ConvTranspose output. Compute spatial sums over input planes, reduce weights over kernel volume, do a tiny channel-mixing GEMM, add bias contribution scaled by output element count, then apply eval BatchNorm to the mean.

```python
input_sums = spatial_sum3d_triton(x)  # [N, C_in], fp32
weight_sums = conv_transpose.weight.to(torch.float32).sum(dim=(2, 3, 4))  # [C_in, C_out]
y_sum = torch.matmul(input_sums, weight_sums) * float(scale_factor)
if conv_transpose.bias is not None:
    y_sum = y_sum + conv_transpose.bias.to(torch.float32) * (float(scale_factor) * out_elems)
mean = y_sum * (1.0 / float(out_elems))

rm = bn.running_mean.to(torch.float32).view(1, -1)
rv = bn.running_var.to(torch.float32).view(1, -1)
affine = torch.rsqrt(rv + bn.eps)
if bn.affine:
    affine = affine * bn.weight.to(torch.float32).view(1, -1)
    bias = bn.bias.to(torch.float32).view(1, -1)
out = (mean - rm) * affine + bias
```

Rationale: global sum is linear. For the simple ConvTranspose geometry, each input value contributes to every kernel coefficient once in the full output sum, so `sum(convT(x,w)) = matmul(sum_spatial(x), sum_kernel(w)) + bias * out_elems`.

## Triton spatial-sum kernel

Use one program per `(N, C_in)` plane for normal shapes, with a persistent fallback when `N*C_in > 65535`:

```python
_MAX_GRID = 65535
_SUM_BLOCK = 2048

if rows <= _MAX_GRID:
    spatial_sum_direct[(rows,)](x_flat, sums, rows, L=L, BLOCK=_SUM_BLOCK)
else:
    spatial_sum_persistent[(_MAX_GRID,)](x_flat, sums, rows, _MAX_GRID, L=L, BLOCK=_SUM_BLOCK)
```

The persistent loop must iterate over rows/tiles, not elements. Unit tests should force this path by temporarily lowering `_MAX_GRID` on a small tensor.

## Correctness/fallback requirements

- Gate the shortcut on all geometry/eval preconditions; otherwise route to the original semantic order (`conv_transpose3d -> scale -> batch_norm -> adaptive_avg_pool3d`).
- Keep all reductions and BN affine math in fp32 before casting back to the input dtype.
- Include benchmark shapes for small, irregular, and default-required inputs.
- If `base_*.py` is read-only or sandboxed, keep `Baseline Triton2` parser-visible with `SKIP_UNAVAILABLE ... max_abs=inf` and `inf` timing cells rather than reading it.

## Cannsim reporting

Cannsim can compare the old global-pool reduction micro-kernel against the new input spatial-sum micro-kernel, but it does not capture the full algorithmic win from avoiding ConvTranspose output materialization. Report both:

- sub-kernel trace deltas (wall cycles, pipeline table, hardware ns), and
- full hardware latency from `remote_verify` for the production path.
