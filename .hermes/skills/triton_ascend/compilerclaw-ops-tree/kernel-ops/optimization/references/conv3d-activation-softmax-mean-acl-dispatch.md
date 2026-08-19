# Conv3d + activation + softmax(C) + spatial mean ACL dispatch

Use this when the main `Conv3d` already runs through ACL/PyTorch and the custom Triton epilogue computes an activation such as HardSwish/ReLU, then `softmax(dim=1)`, then a spatial mean.

## Pattern

Treat ACL post-ops as a first-class production candidate, not only as a reference. A one-program-per-batch Triton epilogue can serialize many spatial tiles inside each program; a two-phase Triton partial-reduction rewrite can expose `N * n_tiles` parallelism, but it adds a second launch and partial GM traffic. Hardware may still favor ACL for multi-tile shapes.

```python
x = self.conv(x).contiguous()
n_tiles = triton.cdiv(D * H * W, BLOCK_S)
if n_tiles > 1 or C > TRITON_MAX_C or n_tiles * N > 65535:
    y = torch.relu(x) * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0
    return torch.softmax(y, dim=1).mean(dim=(2, 3, 4))
return _tiny_triton_postop(x)
```

## Cannsim interpretation

Sub-kernel cannsim on one spatial tile mostly measures the vector softmax body. It will not show full-shape wins or losses from exposing more tile parallelism, extra launch overhead, or ACL dispatch. Use cannsim to verify per-tile bottlenecks, then let remote hardware decide the production threshold.

## Tests and profiling

- Keep all providers parser-visible in `profile_kernels.py`: PyTorch / ACL, baseline1, baseline2, optimized.
- Add a correctness-only tiny single-tile shape if a Triton tiny path remains.
- Add a correctness-only wide-channel or grid-cap shape if the ACL fallback is required for generality.
- If a first hardware verify shows the two-phase Triton candidate slower than ACL, patch dispatch before final reports and ensure `results.txt` reflects the latest observed run.
