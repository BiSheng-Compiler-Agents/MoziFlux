# Row-wise Triplet Margin Loss Pattern

Use this for `TripletMarginLoss`-style kernels over 2D `(B, D)` tensors where each row independently computes two L2 distances and the final output is a scalar mean.

## Recognition signals

- Inputs are `anchor`, `positive`, `negative` with equal 2D shape `(B, D)`.
- The baseline launches one program per row, computes `sqrt(sum((a-p)^2)+eps)` and `sqrt(sum((a-n)^2)+eps)`, stores `loss[B]`, then returns `loss.mean()` from PyTorch.
- Target `B` may approach or exceed Ascend's 65,535 FFTS grid cap.

## Recommended dispatch pattern

Use a three-path host dispatch:

```python
if B <= SMALL_ATOMIC_THRESHOLD:
    # one launch, atomic_add(loss / B) into scalar output
    _row_atomic_kernel[(B,)](...)
elif B <= 65535:
    # preserve fast per-row baseline structure, then out.mean()
    _row_direct_kernel[(B,)](...)
else:
    # legality path for oversized batch
    _row_persistent_kernel[(65535,)](..., n_programs=65535)
```

`SMALL_ATOMIC_THRESHOLD` must be hardware-tuned. A small/medium atomic path can win by removing the second `out.mean()` launch, but it can regress at large `B` from atomic contention; keep the direct row+mean path for large legal batches unless hardware proves otherwise.

## Address-generation caution

Do not assume a contiguous-only rewrite is faster. For this row-wise L2 loss pattern, preserving the baseline row pointer + stride addressing can have a better sub-kernel instruction mix than rewriting every load as `base = row * D; ptr + base + offs`. Test with same-shape cannsim before adopting a contiguous rewrite.

```python
a_row_ptr = anchor_ptr + row * stride_a0
offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
a = tl.load(a_row_ptr + offs * stride_a1, mask=mask, other=0.0)
```

## Persistent path test

Add a correctness-only synthetic shape such as `(B=65536, D=1)` to force the persistent row-loop without large memory. Pre-skip comparison providers whose direct row grid would exceed `coreDim`; do not let them poison the NPU context.
