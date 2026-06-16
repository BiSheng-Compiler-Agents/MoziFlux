import triton
import triton.language as tl


@triton.jit
def _fill_zero_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    # Write zeros safely with mask
    tl.store(out_ptr + offsets, 0.0, mask=mask)
