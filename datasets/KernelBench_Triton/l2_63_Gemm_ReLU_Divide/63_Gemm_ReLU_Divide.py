import triton
import triton.language as tl


@triton.jit
def _relu_divide_inplace_kernel(x_ptr, n_elements, divisor,
                                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x > 0, x / divisor, 0.0)
    tl.store(x_ptr + offsets, x, mask=mask)
