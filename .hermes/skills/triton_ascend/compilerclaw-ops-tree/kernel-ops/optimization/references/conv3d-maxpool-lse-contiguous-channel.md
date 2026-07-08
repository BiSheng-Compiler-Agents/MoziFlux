# Conv3d + MaxPool3d + channel LogSumExp/ReLU: contiguous-channel reduction

Use this reference when a model runs a standard `Conv3d`, standard `MaxPool3d`, then reduces over channel with `logsumexp(dim=1, keepdim=True)` and ReLU.

## Recognition

- Convolution/pooling are standard PyTorch/ACL ops and should usually stay on ACL.
- The custom Triton epilogue reduces over `C` while the tensor is contiguous NCDHW, so channel loads for a fixed spatial point are strided by `D*H*W`.
- A strided-channel LogSumExp kernel may show high PUSHQ / SCALARLDST / SCALAR pressure, or even fail BiSheng with VF stack spill for production-like `C` and `BLOCK` values.

## Optimization pattern

Materialize channel as the innermost dimension once, then run a row-wise contiguous-channel reduction:

```python
x = self.conv(x)
x = self.max_pool(x)
n, c, d, h, w = x.shape
x_last = x.permute(0, 2, 3, 4, 1).contiguous()
return _lse_relu_lastdim_kernel_dispatch(x_last, (n, 1, d, h, w))
```

Device-side kernel sketch:

```python
rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
cols = tl.arange(0, BLOCK_C)
vals = tl.load(x_ptr + rows[:, None] * C + cols[None, :],
               mask=(rows[:, None] < M) & (cols[None, :] < C),
               other=-float("inf")).to(tl.float32)
m = tl.max(vals, axis=1)
s = tl.sum(tl.exp(vals - m[:, None]), axis=1)
y = tl.maximum(tl.log(s) + m, 0.0)
tl.store(out_ptr + rows, y, mask=rows < M)
```

## Cannsim guidance

- If the production baseline cannot compile or simulate because of VF stack spill, use a scale-limited microprobe that preserves the strided-channel reduction structure and report the limitation explicitly.
- Compare trace bottlenecks, not raw cycle counts alone. A useful win is reducing PUSHQ/SCALARLDST dispatch pressure in the strided reduction path.
- The optimized contiguous-channel kernel should be traced at the production channel tile when possible (e.g. `C=64, BLOCK_C=64`) because it is usually compact enough.

## Hardware dispatch caveat

The `permute(...).contiguous()` copy can dominate at medium/default shapes. Even when the contiguous Triton reduction has a better cannsim trace than the strided custom reduction, pure ACL (`conv -> max_pool -> torch.logsumexp -> relu`) may be faster end-to-end on hardware. Keep ACL as a production candidate and let `remote_verify` choose thresholds.

## Profiling requirements

- Keep parser-visible provider columns: `PyTorch / ACL`, `Baseline Triton1`, `Baseline Triton2` if present, and `Optimized Triton`.
- If baseline providers fail MLIR/BiSheng, keep their columns as `inf` / pre-skipped comparison providers; gate success on optimized correctness.
- Report both cannsim trace caveats and physical hardware latency in `performance_report.md`.
