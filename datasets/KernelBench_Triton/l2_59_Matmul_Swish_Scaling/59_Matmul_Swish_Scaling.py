import triton
import triton.language as tl

@triton.jit
def _swish_scale_kernel(x_ptr, y_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Numerically stable sigmoid:
    # sigmoid(x) = 1 / (1 + exp(-x)) for x>=0; = exp(x) / (1 + exp(x)) for x<0
    z = tl.exp(-tl.abs(x))
    s = tl.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))

    out = (x * s) * scale
    tl.store(y_ptr + offs, out, mask=mask)
