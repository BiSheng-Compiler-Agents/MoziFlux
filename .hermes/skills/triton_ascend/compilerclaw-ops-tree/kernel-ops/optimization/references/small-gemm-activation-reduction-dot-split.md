# Small GEMM + activation + row reduction: use Cube dot, split global reduction

## Trigger

Use this pattern for small dense linear layers followed by an elementwise activation, hidden-dimension reduction, and a final batch/global reduction, e.g.:

```python
out = torch.logsumexp(torch.sigmoid(x @ weight.T + bias).sum(dim=1), dim=0)
```

Typical shapes may be too small for a large GEMM kernel (`K≈10-32`, `H≈20-64`) but still benefit from Cube hardware when row tiles are batched.

## Anti-pattern

A single persistent CTA that emulates GEMM with vector broadcast multiply and `tl.sum`:

```python
x_tile = tl.load(...).to(tl.float32)          # [BLOCK_B, BLOCK_K]
w_tile = tl.load(...).to(tl.float32)          # [BLOCK_H, BLOCK_K]
acc += tl.sum(x_tile[:, None, :] * w_tile[None, :, :], axis=2)
```

Cannsim symptoms: PUSHQ/RVECEX/RVECLD/RVECST dominate; Cube is absent or nearly absent.

## Pattern

Split the operator into:

1. A row-tiled Cube kernel: `GEMM + bias + sigmoid + sum_hidden -> row_sums[B]`.
2. A tiny stable logsumexp kernel over `row_sums`.

Core kernel shape:

```python
acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                 mask=(offs_m[:, None] < B) & (offs_k[None, :] < K), other=0.0)
w_tile = tl.load(w_ptr + offs_h[None, :] * stride_wj + offs_k[:, None] * stride_wk,
                 mask=(offs_k[:, None] < K) & (offs_h[None, :] < H), other=0.0)
acc = tl.dot(x_tile, w_tile, acc)
z = acc + bias[None, :]
row_sum += tl.sum(tl.where(row_and_h_mask, tl.sigmoid(z), 0.0), axis=1)
```

Host dispatch:

```python
rows = torch.empty((B,), device=x.device, dtype=torch.float32)
_rowsum_dot_kernel[(triton.cdiv(B, 16),)](..., rows, BLOCK_M=16, BLOCK_H=32, BLOCK_K=16)
_logsumexp_kernel[(1,)](rows, out, B, BLOCK=256)
```

## Why it works

- `tl.dot` activates Cube and removes vector GEMM emulation.
- Row tiling exposes batch parallelism that a single persistent CTA hides.
- Keeping `bias + sigmoid + hidden sum` in the dot kernel avoids materializing `[B,H]` intermediates.
- The extra `[B]` row-sum global write is usually cheaper than doing all GEMM work on vector pipelines.

## Verification

Use cannsim sub-kernel A/B traces:

| Metric to check | Expected direction |
|---|---|
| wall_cycles | down |
| x_events | strongly down |
| PUSHQ / RVECEX / RVECLD / RVECST | strongly down |
| CUBE | appears |
| MTE3 / scalar stalls | may become visible after vector work is removed |

Then run hardware profiling with default, boundary, and aligned shapes. Include both original baseline and read-only `base_*.py` as comparison providers.

## Pitfalls

- Do not cast dot operands to FP32 before `tl.dot`; keep native input dtype and accumulate into FP32.
- Keep `BLOCK_M`, `BLOCK_H`, and `BLOCK_K` multiples of 16 for Cube.
- Every padded `B/K/H` lane needs a mask; small irregular dimensions are the common case.
- For very large `B`, replace the single-program final logsumexp with a two-phase reduction.
