import triton
import triton.language as tl


@triton.jit
def _softplus_kernel(
    x_ptr,  # *const input
    y_ptr,  # *mut output
    n_elements,
    THRESHOLD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # PyTorch F.softplus default behavior (beta=1, threshold=20):
    # if x > threshold: y = x
    # else: y = log1p(exp(x))
    # Use a numerically-stable equivalent for the "else" branch:
    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    cond = x32 > THRESHOLD
    abs_x = tl.abs(x32)
    stable_term = tl.maximum(x32, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    y32 = tl.where(cond, x32, stable_term)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)
