import triton
import triton.language as tl


@triton.jit
def _swish_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    sigmoid = tl.sigmoid(xf)
    y = (xf * sigmoid).to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)
