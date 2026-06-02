import triton
import triton.language as tl

@triton.jit
def _elu_kernel(x_ptr, y_ptr, N, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Help compiler with vectorized/coalesced memory ops
    tl.max_contiguous(offs, 128)

    x = tl.load(x_ptr + offs, mask=mask, other=0)

    # ELU: y = x if x > 0 else alpha * (exp(x) - 1)
    # Use x_neg to avoid exponentiating positive values
    x_neg = tl.minimum(x, 0)
    # Faster exponent: exp(x) = 2^(x / ln(2))
    inv_ln2 = 1.4426950408889634
    exp_term = tl.exp2(x_neg * inv_ln2)
    neg_part = (exp_term - 1) * alpha
    y = tl.where(x > 0, x, neg_part)

    tl.store(y_ptr + offs, y, mask=mask)
