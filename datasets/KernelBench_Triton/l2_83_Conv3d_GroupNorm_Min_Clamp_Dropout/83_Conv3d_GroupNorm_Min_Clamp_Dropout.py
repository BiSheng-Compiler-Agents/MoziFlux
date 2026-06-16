import triton
import triton.language as tl


@triton.jit
def _fill_const_kernel(out_ptr, value, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, value, mask=mask)
