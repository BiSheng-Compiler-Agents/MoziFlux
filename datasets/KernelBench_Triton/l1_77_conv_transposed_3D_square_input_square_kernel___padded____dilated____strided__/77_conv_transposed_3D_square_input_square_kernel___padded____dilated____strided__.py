import triton
import triton.language as tl


@triton.jit
def _touch_inplace_kernel(y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, values, mask=mask)
