# ConvTranspose + Pool/Reduction Hybrid Dispatch

Use this when a KernelBench-style operator already uses ACL for `ConvTranspose2d` but replaces the post-processing chain (`MaxPool2d`/activation/mean/tanh/etc.) with a custom Triton reduction.

## Recognition pattern

- The convolution/transposed convolution itself is a mature ACL `nn.ConvTranspose*` module.
- The editable baseline calls ACL convolution, then launches a custom Triton kernel for pooling + activation + spatial reduction.
- The custom post kernel uses one large vector per `(B,C)` plane or a grid-capped persistent decomposition for target shapes.
- Cannsim sub-kernel improvements are modest and hardware timing shows custom persistent post-processing loses to ACL for medium/large spatial planes.

## Preferred optimization

Use a hybrid dispatch instead of forcing all shapes through Triton:

```python
x = self.conv_transpose(x).contiguous()
B, C, H, W = x.shape
H_OUT = H // 2
W_OUT = W // 2
TOT = H_OUT * W_OUT
n_tiles_per_bc = triton.cdiv(TOT, BLOCK)
total_tiles = B * C * n_tiles_per_bc

# Keep only the measured tiny-plane Triton fast path.
if TOT > TINY_THRESHOLD or total_tiles > 65535:
    y = self.maxpool(x)
    y = self.hardtanh(y)
    y = y.mean(dim=(2, 3), keepdim=True)
    return torch.tanh(y)

# Tiny path: fused Triton maxpool/hardtanh/reduction/tanh.
```

Important: when falling back after `x = self.conv_transpose(x)`, do **not** call a helper that performs `conv_transpose` again. Apply only the remaining post-processing ops to the already-computed tensor.

## Cannsim interpretation

A bounded probe can compare instruction-level changes such as replacing exp/div tanh with `tl.math.tanh`, but it cannot prove that the full persistent post-processing path beats ACL. Treat cannsim as diagnostic; let `remote_verify` decide the dispatch threshold.

Report both facts honestly:

| Evidence | How to use it |
|---|---|
| Cannsim bounded probe improves wall cycles | Keep instruction-level fix for tiny/custom path |
| Hardware persistent path is slower than ACL | Route medium/target planes to ACL |
| Baseline full custom path is toxic/huge | Pre-skip baseline provider as `inf` in profiling, keep column visible |

## Profiling requirements

- Include a tiny shape that triggers the Triton fast path.
- Include a medium/target shape that triggers ACL fallback.
- Correctness-check optimized output on both paths.
- If a read-only `base_*.py` or huge baseline comparison provider is toxic, pre-skip it neutrally (`SKIP`/`inf`) rather than poisoning the NPU context.
