import triton
import triton.language as tl

@triton.jit
def _log_softmax_row_fused_kernel(x_ptr, y_ptr, D, BLOCK_SIZE: tl.constexpr):
    """
    One program processes one row (length D) and computes log_softmax in a single pass
    using registers:
      y = x - (m + log(sum(exp(x - m))))
    """
    pid = tl.program_id(axis=0)
    row_start = pid * D
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Load row and upcast to fp32 for stable reductions
    x = tl.load(x_ptr + row_start + offs, mask=mask, other=-float("inf"))
    x32 = x.to(tl.float32)

    # Stable log-softmax: subtract row max, sum exp, subtract log-sum
    m = tl.max(x32, axis=0)
    x_shift = x32 - m
    exp_x = tl.exp(x_shift)
    denom = tl.sum(exp_x, axis=0)
    log_denom = tl.log(denom)
    y = x_shift - log_denom

    # Store back (implicit cast to output dtype)
    tl.store(y_ptr + row_start + offs, y, mask=mask)
