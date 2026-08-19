# Optimizations Applied

## 1. Replaced vector-emulated GEMM with Cube `tl.dot`

Baseline performed each hidden tile as vector broadcast multiply plus `tl.sum`:

```python
w_tile = tl.load(w_ptrs, mask=h_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
z += tl.sum(w_tile * x_vals[None, :], axis=1)
```

Optimized code tiles `(B, H, K)` and accumulates with Cube hardware:

```python
acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
for k0 in tl.range(0, I, BLOCK_K):
    x = tl.load(x_ptr + rows[:, None] * stride_xb + ks[None, :] * stride_xi, mask=...)
    w = tl.load(w_ptr + hs[None, :] * stride_wh + ks[:, None] * stride_wi, mask=...)
    acc = tl.dot(x, w, acc)
```

Rationale: `tl.dot` is the only Triton-Ascend path that activates Cube. This batches 16 input rows per program and removes the baseline's scalar/vector inner-product loop.

## 2. Split GEMM and sigmoid-sum into two kernels

A direct fused post-dot `sigmoid(acc + bias)` in the Cube kernel triggered a BiSheng `memref.expand_shape` compile failure during cannsim compilation. The optimized implementation now materializes only the logits matrix and performs the bias/sigmoid/row reduction in a vector kernel:

```python
logits = torch.empty((B, H), device=x.device, dtype=torch.float32)
_matmul_logits_kernel[grid_dot](... logits, ...)
_sigmoid_sum_kernel[(B,)](logits, bias_in, out, B, H, stride_bo, BLOCK_H=1024)
```

Rationale: the intermediate is `B*H` fp32 (16 MiB for the default shape), far cheaper than emulating the full `B*H*I` multiply on Vector pipelines, and it keeps both dispatch paths compiler-legal.

## 3. Persistent 1D grid capped to physical AI Cores

Instead of launching one program per `(B,H)` tile directly, the dot kernel uses a 1D grid over physical AI Cores and loops over tiles internally:

```python
num_aicore = _device_prop("num_aicore", 20)
grid_dot = (max(1, min(num_aicore, _MAX_GRID, total_tiles)),)
for tile_id in tl.range(pid, total_tiles, tl.num_programs(0)):
    ...
```

Rationale: this avoids multidimensional/grid-overflow issues and balances the 4096 default tiles across AI Cores.

## 4. Boundary-safe masks and contiguous layout

All GM loads/stores are masked, and host inputs are made contiguous before launch:

```python
x_in = x.contiguous()
weight_in = weight.contiguous()
mask=row_mask[:, None] & k_mask[None, :]
```

Rationale: Ascend has zero tolerance for OOB access. Contiguous `x`, `weight`, and `logits` also let MTE form larger transfers than arbitrary strided input layouts.
