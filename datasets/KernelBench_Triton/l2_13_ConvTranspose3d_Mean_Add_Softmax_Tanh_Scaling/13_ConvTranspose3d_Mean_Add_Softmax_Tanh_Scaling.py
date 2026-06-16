import triton
import triton.language as tl


@triton.jit
def _fill_const_kernel(out_ptr, value, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Hints for better vectorization/coalescing
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, 16)
    # Store the scalar constant; Triton will broadcast and cast to dst dtype
    tl.store(out_ptr + offsets, value, mask=mask)
