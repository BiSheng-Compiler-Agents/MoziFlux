import triton
import triton.language as tl


@triton.jit
def _touch_tensor_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    _ = tl.load(x_ptr + offs, mask=mask, other=0.0)
