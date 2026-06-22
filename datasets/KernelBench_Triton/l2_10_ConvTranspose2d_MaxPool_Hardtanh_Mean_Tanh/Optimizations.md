# Optimizations

## 1. Hybrid Triton/ACL dispatch

The source uses ACL `ConvTranspose2d`, then launches a custom Triton kernel for maxpool, hardtanh, mean and tanh. Hardware timing showed the custom fused path wins only for tiny spatial planes, while ACL is much faster for medium/target planes.

```python
if TOT > 64 or total_tiles > _MAX_PROGRAMS:
    y = self.maxpool(x)
    y = self.hardtanh(y)
    y = y.mean(dim=(2, 3), keepdim=True)
    return torch.tanh(y)
```

This keeps the fastest tiny-shape fused path and routes the original benchmark shape to mature ACL kernels.

## 2. UB-safe tiled spatial reduction for the Triton fast path

Baseline reduces the full pooled plane in one program:

```python
BLOCK_W = next_pow2(H_OUT * W_OUT)
_fused_maxpool2x2_hardtanh_mean_tanh[(B * C,)](..., BLOCK_W=BLOCK_W)
```

The optimized Triton path splits work into 1024-element tiles and fp32 partial sums:

```python
n_tiles_per_bc = triton.cdiv(TOT, _BLOCK_ELEMS)
partial = torch.empty((B * C, n_tiles_per_bc), device=x.device, dtype=torch.float32)
```

This avoids oversized live vectors when the Triton path is used.

## 3. Grid-cap aware direct/persistent structure

The optimized custom path contains both direct and persistent variants:

```python
if total_tiles <= _MAX_PROGRAMS:
    _partial_maxpool2x2_hardtanh_sum_direct[(total_tiles,)](...)
else:
    _partial_maxpool2x2_hardtanh_sum_persistent[(_MAX_PROGRAMS,)](...)
```

Production currently routes large planes to ACL before persistent launch because hardware showed persistent custom reduction is slower than ACL on the target.

## 4. Hardware tanh instruction

Baseline computes tanh via exp/div:

```python
e2 = tl.exp(2.0 * mean_val)
out_val = 1.0 - 2.0 / (e2 + 1.0)
```

The optimized finalize kernel uses Ascend's vector tanh lowering:

```python
y = tl_math.tanh(total * (1.0 / TOT))
```

Cannsim: 15,329 -> 14,014 wall cycles on the bounded probe, mainly from PUSHQ reduction.
