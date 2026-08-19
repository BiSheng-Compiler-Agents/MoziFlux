# ConvTranspose3d + Softmax(dim=1) + Sigmoid Hybrid Dispatch

Use this reference when a KernelBench-style operator keeps `nn.ConvTranspose3d` on ACL/PyTorch and implements the post-processing chain `softmax(dim=1)` followed by `sigmoid` in a custom Triton epilogue.

## Recognition pattern

- The convolution is already `nn.ConvTranspose3d`; do not rewrite it as a custom convolution unless explicitly required.
- The Triton epilogue maps one program to one `(N, D, H, W)` row and reduces over channel `C`.
- Default 3D volumes produce `total_rows = N * D_out * H_out * W_out`, which can exceed Ascend FFTS `coreDim <= 65535`.
- The channel count is modest (often `C=64`), so a single row fits in UB and can be computed in one pass for small tensors.

## Preferred optimization

Keep a single-pass Triton epilogue for small/legal row counts, and route large/default tensors to native ACL post-processing:

```python
_MAX_PROGRAMS = 65535
_TRITON_MAX_C = 128

x = self.conv_transpose(x)
N, C, D, H, W = x.shape
total_rows = N * D * H * W

if total_rows > _MAX_PROGRAMS or C > _TRITON_MAX_C:
    return torch.sigmoid(torch.softmax(x, dim=1))

return _singlepass_softmax_sigmoid_tiny(x)
```

For the tiny Triton path, replace the baseline three-pass row loop (max pass, denominator pass, store pass) with a single UB-resident load:

```python
xv = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
m = tl.max(xv, axis=0)
e = tl.exp(xv - m)
soft = e / tl.sum(e, axis=0)
tl.store(out_ptrs, 1.0 / (1.0 + tl.exp(-soft)), mask=mask)
```

## Profiling requirements

- Include a tiny/small shape below the grid cap to exercise the Triton direct path.
- Include the exact/default shape to exercise ACL dispatch.
- Keep `Baseline Triton1` and `Baseline Triton2` columns visible, but pre-skip their default-shape timing if `total_rows > 65535`; launching them can poison the NPU context.
- Unit-test the optimized path on both dispatch routes.

## Cannsim interpretation

Cannsim can compare the small Triton epilogue sub-kernel only. It cannot measure the full-shape win/legality of routing to ACL, so report ACL/default performance from `remote_verify` hardware results and do not fabricate cannsim traces for removed Triton work.
