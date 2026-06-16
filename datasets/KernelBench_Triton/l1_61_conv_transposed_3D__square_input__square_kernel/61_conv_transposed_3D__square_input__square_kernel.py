import triton
import triton.language as tl


@triton.jit
def _copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, values, mask=mask)
