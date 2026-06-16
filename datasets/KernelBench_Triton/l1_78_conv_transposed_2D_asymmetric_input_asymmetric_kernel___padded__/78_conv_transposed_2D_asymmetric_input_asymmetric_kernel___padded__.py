import triton
import triton.language as tl


@triton.jit
def _touch_tensor_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    _ = tl.load(x_ptr + offsets, mask=mask, other=0.0)
