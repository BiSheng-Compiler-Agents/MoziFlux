import triton
import triton.language as tl

@triton.jit
def _gelu_fwd_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    x3 = x32 * x32 * x32
    inner = 0.7978845608028654 * (x32 + 0.044715 * x3)
    abs_inner = tl.abs(inner)
    exp_term = tl.exp(-2.0 * abs_inner)
    tanh_abs = (1.0 - exp_term) / (1.0 + exp_term)
    tanh_inner = tl.where(inner >= 0.0, tanh_abs, -tanh_abs)
    y = (0.5 * x32 * (1.0 + tanh_inner)).to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)
