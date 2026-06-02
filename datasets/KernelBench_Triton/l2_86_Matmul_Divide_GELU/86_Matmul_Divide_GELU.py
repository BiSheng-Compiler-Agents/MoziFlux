import triton
import triton.language as tl

@triton.jit
def fused_div_gelu_kernel(x_ptr, out_ptr, n_elements, inv_divisor, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    y = x32 * inv_divisor

    y3 = y * y * y
    inner = 0.7978845608028654 * (y + 0.044715 * y3)
    abs_inner = tl.abs(inner)
    exp_term = tl.exp(-2.0 * abs_inner)
    tanh_abs = (1.0 - exp_term) / (1.0 + exp_term)
    tanh_inner = tl.where(inner >= 0.0, tanh_abs, -tanh_abs)
    out = (0.5 * y * (1.0 + tanh_inner)).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)
