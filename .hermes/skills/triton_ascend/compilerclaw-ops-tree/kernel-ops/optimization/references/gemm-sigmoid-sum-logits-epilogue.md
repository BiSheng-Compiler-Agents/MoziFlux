# Large GEMM + sigmoid + hidden-sum: Cube logits plus vector epilogue

## Trigger

Use this pattern for operators shaped like:

```python
out = torch.sigmoid(x @ weight.T + bias).sum(dim=1, keepdim=True)
```

especially when `B`, `I/K`, and `H` are large enough that vector-emulated inner products dominate runtime.

## Anti-pattern

A fused single kernel that computes every hidden output with vector broadcast multiply and `tl.sum`:

```python
x_vals = tl.load(...).to(tl.float32)             # [BLOCK_K]
w_tile = tl.load(...).to(tl.float32)             # [BLOCK_H, BLOCK_K]
z += tl.sum(w_tile * x_vals[None, :], axis=1)
```

Cannsim symptoms: SCALAR/PUSHQ/RVECEX dominate and CUBE is absent.

## Pattern

Split into two compiler-legal kernels:

1. A persistent 1D-grid Cube kernel computes fp32 logits `[B, H]` with `tl.dot`.
2. A row-wise vector kernel applies bias + sigmoid and reduces each row.

```python
acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
for k0 in tl.range(0, I, BLOCK_K):
    x = tl.load(..., mask=row_mask[:, None] & k_mask[None, :], other=0.0)
    w = tl.load(..., mask=k_mask[:, None] & h_mask[None, :], other=0.0)
    acc = tl.dot(x, w, acc)
tl.store(logits_ptr + rows[:, None] * H + hs[None, :], acc,
         mask=row_mask[:, None] & h_mask[None, :])

# second kernel, one program per row
total += tl.where(hs < H, 1.0 / (1.0 + tl.exp(-(z + b))), 0.0)
tl.store(out_ptr + row, tl.sum(total, axis=0), mask=row < B)
```

Host dispatch:

```python
logits = torch.empty((B, H), device=x.device, dtype=torch.float32)
grid_dot = (max(1, min(num_aicore, 65535, cdiv(B, BLOCK_M) * cdiv(H, BLOCK_H))),)
_matmul_logits_kernel[grid_dot](x, weight, logits, ...)
_sigmoid_sum_kernel[(B,)](logits, bias, out, B, H, BLOCK_H=1024)
```

## Why it works

- `tl.dot` activates Cube and removes vector GEMM emulation.
- Materializing `[B,H]` fp32 logits is usually much cheaper than doing `B*H*K` multiply-add work on vector pipelines.
- A persistent 1D grid capped to physical AI Cores avoids multidimensional grid and FFTS overflow issues.
- The vector epilogue is simple, boundary-safe, and easy to test across irregular and large shapes.

## Compiler pitfall

Trying to fully fuse post-dot `bias + sigmoid + sum` directly in the Cube kernel can hit a BiSheng `memref.expand_shape` error when broadcasting a loaded bias vector across the accumulator tile. If that happens, use the two-kernel logits + vector epilogue split above.

## Verification

Compare cannsim sub-kernels using normalized work, not raw wall cycles, if the optimized tile computes more rows than baseline.

| Metric | Expected direction |
|---|---|
| normalized cycles / row | strongly down |
| RVECEX / PUSHQ from GEMM | strongly down |
| CUBE | appears |
| MTE3 | may rise because logits are materialized |

Run hardware profiling with PyTorch/ACL, editable baseline, read-only baseline, and optimized provider. Unit tests should include irregular, aligned, large-K, and default-sized shapes.