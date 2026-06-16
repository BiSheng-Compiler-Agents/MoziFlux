import triton
import triton.language as tl


@triton.jit
def _fill_ones_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    # Hints to help the compiler generate efficient, coalesced stores
    tl.multiple_of(offs, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    # Use a vector register of ones to encourage wide stores
    ones = tl.full([BLOCK_SIZE], 1.0, tl.float32)
    tl.store(out_ptr + offs, ones, mask=mask)
