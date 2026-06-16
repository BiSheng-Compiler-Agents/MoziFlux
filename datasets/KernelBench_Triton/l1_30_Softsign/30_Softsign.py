import triton
import triton.language as tl


@triton.jit
def _softsign_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Compute in input dtype to avoid unnecessary upcasts
    one = tl.full([1], 1.0, dtype=x.dtype)
    y = x / (tl.abs(x) + one)
    tl.store(y_ptr + offsets, y, mask=mask)
