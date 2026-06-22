# Optimizations

## 1. Preserve ACL ConvTranspose3d and BatchNorm3d

The expensive stages are mature ACL-covered operators, so the optimized module keeps the baseline construction and dispatch:

```python
x = self.conv_transpose(x)
x = self.batch_norm(x)
```

This avoids replacing ConvTranspose3d with a scalar Triton direct-convolution implementation that would not use Cube effectively.

## 2. Hybrid spatial mean-subtract epilogue

The baseline always launches one Triton program per `(N,C)` plane and serially scans `D*H*W` twice:

```python
while i < S:
    vals = tl.load(base_x + idx, mask=mask, other=0.0).to(tl.float32)
    sum_acc += tl.sum(vals, axis=0)
    i += BLOCK_SIZE
# second pass: store vals - mean
```

The final optimized dispatch keeps this single-launch Triton path only for tiny planes, where launch overhead dominates:

```python
if S <= _BLOCK * 2 and planes <= _MAX_PROGRAMS:
    y = torch.empty_like(x_contig)
    _direct_mean_subtract_kernel[(planes,)](x_contig, y, S, planes, BLOCK=_BLOCK)
    return y
```

For medium and target shapes, hardware showed the custom multi-kernel partial-reduction attempt was much slower than ACL reduction, so the final optimized path uses the mature reduction primitive:

```python
return x_contig - x_contig.mean(dim=(2, 3, 4), keepdim=True)
```

## 3. Hardware-driven rollback of custom partial reduction

A three-kernel Triton epilogue (partial sums, finalizer, subtract) was implemented and tested, but `remote_verify` measured it at `14.909141 ms` on the target versus `1.886290 ms` for baseline Triton1. The final code therefore rolls back production dispatch to ACL mean for `S > 4096`, while retaining the tiny-plane Triton fast path that measured fastest on `small_direct`.

## 4. Verified dispatch results

`profile_kernels.py` covers both final dispatch regimes:

- `small_direct`: optimized Triton direct path, `0.015564 ms`.
- `medium_acl`: ACL mean path, `0.062480 ms`.
- `target`: ACL mean path, `1.723299 ms`.

All optimized paths passed correctness against PyTorch/ACL with max diff `<= 2.98023e-08` on profiled shapes.
