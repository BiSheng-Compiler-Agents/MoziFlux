import triton
import triton.language as tl


@triton.jit
def _tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    abs_x = tl.abs(x32)
    exp_term = tl.exp(-2.0 * abs_x)
    tanh_abs = (1.0 - exp_term) / (1.0 + exp_term)
    y32 = tl.where(x32 >= 0.0, tanh_abs, -tanh_abs)
    y = y32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)
