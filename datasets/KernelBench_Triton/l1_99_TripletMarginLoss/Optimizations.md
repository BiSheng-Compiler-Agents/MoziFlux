# Optimizations: TripletMarginLoss

## 1. Fuse row loss and mean for normal row counts

Baseline computes one row loss per program, stores `out[B]`, then returns `out.mean()` as a second NPU reduction launch:

```python
_triplet_margin_row_kernel[(B,)](..., out, ...)
return out.mean()
```

The optimized path routes `B <= 4096` to a scalar-output row kernel. Each row computes the same fp32 L2 distances and atomically contributes `loss / B`, eliminating the row-output tensor materialization and the follow-up mean launch on small/medium batches where hardware timing shows the atomic path wins.

```python
out = torch.empty((1,), device=a.device, dtype=torch.float32)
out.zero_()
_triplet_margin_row_atomic_kernel[(B,)](..., out, B, D, ..., BLOCK_SIZE=BLOCK_SIZE, N_ITERS=N_ITERS)
return out[0]
```

## 2. Preserve the baseline address-generation pattern in the atomic path

A contiguous-only row-address rewrite was tested in cannsim but regressed the sub-kernel trace because PUSHQ/MTE wait increased. The shipped atomic path keeps the baseline's row pointer plus stride-offset addressing, which preserves the faster per-row instruction mix while changing only the final accumulation strategy.

```python
a_row_ptr = anchor_ptr + row * stride_a0
offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
a = tl.load(a_row_ptr + offs * stride_a1, mask=mask, other=0.0)
```

## 3. Add a grid-capped persistent row path for oversized batches

The editable baseline launches `grid=(B,)`, which is illegal when `B > 65535` on Ascend. The optimized host dispatch uses the atomic direct path up to the grid cap, then uses a persistent row loop for oversized batches.

```python
if B <= 4096:
    _triplet_margin_row_atomic_kernel[(B,)](...)
elif B <= 65535:
    _triplet_margin_row_direct_kernel[(B,)](...)
else:
    _triplet_margin_row_persistent_kernel[(65535,)](..., n_programs=65535)
```

Inside the persistent kernel, each program advances by `n_programs` over rows, preserving correctness for arbitrary valid 2D inputs.
