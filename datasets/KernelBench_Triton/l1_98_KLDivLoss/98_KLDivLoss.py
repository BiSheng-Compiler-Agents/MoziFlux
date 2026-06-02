import triton
import triton.language as tl

@triton.jit
def _kl_div_batch_sum_kernel(
    pred_ptr, targ_ptr, out_ptr,
    B, D,
    stride_pb, stride_pd,
    stride_tb, stride_td,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, BLOCK_SIZE)

    # Base pointers for the current row
    pred_row_ptr = pred_ptr + row * stride_pb
    targ_row_ptr = targ_ptr + row * stride_tb

    # Accumulate per-lane and reduce once at the end
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    n_iters = tl.cdiv(D, BLOCK_SIZE)

    for k in range(0, n_iters, 2):
        # First tile
        cols0 = k * BLOCK_SIZE + offs
        mask0 = cols0 < D
        p0 = tl.load(pred_row_ptr + cols0 * stride_pd, mask=mask0, other=1.0).to(tl.float32)
        t0 = tl.load(targ_row_ptr + cols0 * stride_td, mask=mask0, other=0.0).to(tl.float32)

        # Safe t0 * log(t0): define 0*log(0)=0
        t0_pos = t0 > 0.0
        t0_safe = tl.where(t0_pos, t0, 1.0)
        t0_logt = tl.where(t0_pos, t0 * tl.log(t0_safe), 0.0)

        contrib0 = t0_logt - t0 * tl.log(p0)
        acc += contrib0

        # Optional second tile (unrolled) to reduce loop overhead
        k1 = k + 1
        if k1 < n_iters:
            cols1 = k1 * BLOCK_SIZE + offs
            mask1 = cols1 < D
            p1 = tl.load(pred_row_ptr + cols1 * stride_pd, mask=mask1, other=1.0).to(tl.float32)
            t1 = tl.load(targ_row_ptr + cols1 * stride_td, mask=mask1, other=0.0).to(tl.float32)

            t1_pos = t1 > 0.0
            t1_safe = tl.where(t1_pos, t1, 1.0)
            t1_logt = tl.where(t1_pos, t1 * tl.log(t1_safe), 0.0)

            contrib1 = t1_logt - t1 * tl.log(p1)
            acc += contrib1

    # Reduce once per row and store (one store per row; no global contention)
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, total)
