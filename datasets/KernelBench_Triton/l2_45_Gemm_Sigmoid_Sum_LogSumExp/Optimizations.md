# Optimizations

## 1. Replace scalar/vector GEMM emulation with `tl.dot`

Baseline computed each row/hidden tile as an elementwise broadcast multiply followed by a vector reduction:

```python
x_tile = tl.load(...).to(tl.float32)          # [BLOCK_B, BLOCK_K]
w_tile = tl.load(...).to(tl.float32)          # [BLOCK_H, BLOCK_K]
acc += tl.sum(x_tile[:, None, :] * w_tile[None, :, :], axis=2)
```

Optimized code maps the small GEMM tile to Cube hardware with native FP32 accumulation:

```python
x_tile = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                 mask=m_mask[:, None] & k_mask[None, :], other=0.0)
w_tile = tl.load(w_ptr + offs_h[None, :] * stride_wj + offs_k[:, None] * stride_wk,
                 mask=k_mask[:, None] & h_mask[None, :], other=0.0)
acc = tl.dot(x_tile, w_tile, acc)
```

Rationale: the operator is GEMM-dominated even though dimensions are small (`B=128,K=10,H=20`). `tl.dot` activates the Cube pipeline and removes thousands of RVEC multiply/add/reduction instructions from the traced sub-kernel.

## 2. Parallelize over row tiles instead of one persistent CTA

Baseline fused all rows into one program:

```python
_fused_rowsum_logsumexp_kernel[(1,)](..., BLOCK_B=32, BLOCK_H=32, BLOCK_K=16)
```

Optimized code computes row sums in independent row tiles:

```python
block_m = 16
grid_rows = (triton.cdiv(B, block_m),)
_gemm_sigmoid_rowsum_dot_kernel[grid_rows](..., BLOCK_M=16, BLOCK_H=32, BLOCK_K=16)
```

Rationale: row-sum computation is independent across `B`, so row tiling exposes parallelism. The final logsumexp is kept as a separate small reduction over the row buffer.

## 3. Preserve fusion where it matters: GEMM + bias + sigmoid + hidden sum

Optimized row kernel keeps the hidden-dimension epilogue inside the same kernel:

```python
bias = tl.load(b_ptr + offs_h * stride_b, mask=h_mask, other=0.0).to(tl.float32)
z = acc + bias[None, :]
sig = tl.sigmoid(z)
sig = tl.where(m_mask[:, None] & h_mask[None, :], sig, 0.0)
row_sum += tl.sum(sig, axis=1)
```

Rationale: this avoids materializing the `[B,H]` linear output and the sigmoid output in global memory. Only one `[B]` FP32 row-sum buffer is written before the final logsumexp.

## 4. Boundary-safe masks on every memory access

All loads and stores use masks:

```python
tl.load(..., mask=m_mask[:, None] & k_mask[None, :], other=0.0)
tl.load(..., mask=k_mask[:, None] & h_mask[None, :], other=0.0)
tl.store(row_ptr + offs_m, row_sum, mask=m_mask)
```

Rationale: the benchmark has non-multiple dimensions (`K=10,H=20`), so all tiles include padding lanes. Masks preserve correctness for default and boundary profile shapes.
