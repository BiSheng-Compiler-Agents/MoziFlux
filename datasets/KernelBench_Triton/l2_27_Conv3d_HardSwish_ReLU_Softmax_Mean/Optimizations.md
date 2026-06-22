# Optimizations

## 1. Hardware-selected ACL dispatch for multi-tile post-op regimes

**Baseline:** one Triton program handles one batch item and iterates over all spatial tiles serially:

```python
grid = (N,)
while s_start < S:
    ...
    acc += tl.sum(p, axis=1)
```

**Candidate Triton optimization:** each `(N, spatial_tile)` program computes channel-softmax partial sums, then a second kernel reduces the small partial buffer:

```python
grid_part = (N * n_tiles,)
_hsrelu_softmax_partials_kernel[grid_part](...)
_partials_reduce_kernel[(N * C,)](...)
```

Hardware profiling showed the candidate was only competitive at the target and slower at medium scale, while the mature ACL expression was fastest for all benchmarked multi-tile regimes. The production dispatch therefore uses ACL when `n_tiles > 1` and keeps the Triton path only for tiny single-tile cases.

## 2. Keep Conv3d on ACL and fuse only the post-op math

```python
x = self.conv(x)
return fused_hswish_relu_softmax_mean(x)
```

Rationale: `Conv3d` is a mature ACL-covered operator; replacing it with a direct Triton convolution would be structurally risky and slower. The optimized kernel focuses on the custom HardSwish/ReLU/Softmax/Mean epilogue where the editable baseline already used Triton.

## 3. Add safe dispatch coverage for tiny Triton and wide-channel ACL paths

```python
if n_tiles > 1 or C > _TRITON_MAX_C or (N * n_tiles) > _MAX_PROGRAMS:
    return _post_ops_acl(x)
```

Rationale: the Triton tile computes softmax across all channels in UB and remains available for the tiny `S=1` single-tile case. Multi-tile, wide-channel, and >65,535-tile cases route to ACL to preserve performance and avoid Ascend `coreDim`/UB hazards; `profile_kernels.py` covers `tiny_triton` and `fallback_c80` correctness-only dispatch tests.
