# Optimizations Applied

## 1. Move channel reduction to contiguous last dimension

**Baseline pattern** reduces channel `C` while channel stride is `Z*Y*X`, so each channel load for a fixed output position jumps by a large spatial plane:

```python
base_in = n64 * sN + z64 * sZ + y64 * sY + x64
p = x_ptr + base_in
val_next = tl.load(p + c * sC, mask=mask, other=-float("inf"))
```

**Optimized pattern** keeps ACL Conv3d and MaxPool3d, then materializes `C` as the innermost dimension once:

```python
x = self.conv(x)
x = self.max_pool(x)
x_last = x.permute(0, 2, 3, 4, 1).contiguous()
return _lse_relu_triton_lastdim(x_last, (n, 1, d, h, w))
```

Rationale: the Triton kernel now loads `[BLOCK_M, BLOCK_C]` contiguous rows, replacing strided channel gathers with coalesced MTE-friendly loads.

## 2. Vectorize LogSumExp across channels in one program

```python
rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
cols = tl.arange(0, BLOCK_C)
vals = tl.load(x_ptr + rows[:, None] * C + cols[None, :],
               mask=(rows[:, None] < M) & (cols[None, :] < C), other=-float("inf"))
m = tl.max(vals.to(tl.float32), axis=1)
s = tl.sum(tl.exp(vals.to(tl.float32) - m[:, None]), axis=1)
out = tl.maximum(tl.log(s) + m, 0.0)
```

Rationale: C=64 fits safely in UB, so a single-pass row-wise reduction avoids the baseline loop-carried scalar channel pointer and exposes a compact vector reduction.

## 3. Preserve numerical stability and precision

The optimized kernel upcasts to fp32 before `max`, `exp`, `sum`, and `log`, then casts back to the pooled tensor dtype at the host boundary:

```python
vals = vals.to(tl.float32)
...
return y_flat.reshape(out_shape).to(x_nhwc.dtype)
```

Remote correctness against PyTorch/ACL passed on `small`, `medium`, and `default`; maximum absolute error was `4.76837e-07`.
