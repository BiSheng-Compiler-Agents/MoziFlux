import triton
import triton.language as tl

@triton.jit
def _cross_entropy_rowwise_kernel(
    x_ptr,             # *f32 [N, C]
    t_ptr,             # *i64 [N]
    out_ptr,           # *f32 [N]
    stride_x_batch,    # int
    stride_x_class,    # int
    N,                 # int
    C,                 # int
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # row id
    row_in_bounds = pid < N

    # Pointers to the start of this row
    row_ptr = x_ptr + pid * stride_x_batch

    # Load logits of this row
    offs = tl.arange(0, BLOCK_SIZE)
    mask_cls = offs < C
    mask = mask_cls & row_in_bounds

    x = tl.load(row_ptr + offs * stride_x_class, mask=mask, other=-float("inf"))

    # numerically stable log-sum-exp
    m = tl.max(x, axis=0)
    x_shift = x - m
    expx = tl.exp(x_shift)
    sumexp = tl.sum(expx, axis=0)
    logsumexp = tl.log(sumexp) + m

    # Load target index and gather target logit
    tgt = tl.load(t_ptr + pid, mask=row_in_bounds, other=0)
    # ensure index dtype for address arithmetic
    tgt = tgt.to(tl.int64)
    x_t = tl.load(row_ptr + tgt * stride_x_class, mask=row_in_bounds, other=0.0)

    # per-sample negative log-likelihood
    nll = logsumexp - x_t

    # Write output
    tl.store(out_ptr + pid, nll, mask=row_in_bounds)
