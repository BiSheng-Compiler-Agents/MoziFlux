import triton
import triton.language as tl

@triton.jit
def _act_softplus_tanh_mul_kernel(x_ptr, y_ptr, n_elements, THRESHOLD: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Compute y = x * tanh(softplus(x)) elementwise with PyTorch's softplus default (beta=1.0, threshold=20.0).
    For x <= THRESHOLD:
        tanh(softplus(x)) = 1 - 2 / (e^{2x} + 2 e^{x} + 2)
    For x > THRESHOLD (softplus(x) = x in PyTorch): use tanh(softplus(x)) ~ 1.0 in fp32.
    This uses a single exp per element and avoids tl.tanh/log to reduce compute.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = x_in.to(tl.float32)

    use_large = x > THRESHOLD
    # Only compute exp for the "small" branch; set x_small=0 for large branch to keep exp well-conditioned.
    x_small = tl.where(use_large, 0.0, x)
    t = tl.exp(x_small)
    den = t * t + 2.0 * t + 2.0
    tanh_small = 1.0 - 2.0 / den

    tval = tl.where(use_large, 1.0, tanh_small)
    y = (tval * x).to(x_in.dtype)

    tl.store(y_ptr + offs, y, mask=mask)
