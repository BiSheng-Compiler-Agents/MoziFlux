import triton
import triton.language as tl


@triton.jit
def _gelu_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Constants for GELU approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    ca = 0.035677408136300125  # c * 0.044715

    # u = sqrt(2/pi) * (x + 0.044715 * x^3) = x * (c + ca * x^2)
    x2 = x_f32 * x_f32
    u = x_f32 * (c + ca * x2)

    # Use identity: 0.5*x*(1 + tanh(u)) == x * sigmoid(2u)
    s = 1.0 / (1.0 + tl.exp(-2.0 * u))
    y_f32 = x_f32 * s
    y = y_f32.to(x.dtype)

    tl.store(y_ptr + offs, y, mask=mask)
