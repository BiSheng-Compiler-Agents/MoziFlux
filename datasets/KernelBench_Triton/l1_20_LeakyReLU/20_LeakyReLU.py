import triton
import triton.language as tl

@triton.jit
def _leaky_relu_kernel(x_ptr, y_ptr, n_elements, neg, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, 16)

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)
    y = tl.where(x >= zero, x, x * neg)
    tl.store(y_ptr + offsets, y, mask=mask)
