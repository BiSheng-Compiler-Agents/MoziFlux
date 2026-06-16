import triton
import triton.language as tl


@triton.jit
def _fused_bias_div_swish_flat_kernel(
    x_ptr,  # *float32, flattened [M*N]
    y_ptr,  # *float32, flattened [M*N] (can alias x_ptr for in-place)
    bias_ptr,  # *float32, shape (1,) scalar bias
    inv_div,  # float32 scalar = 1.0 / divide_value
    N_ELEMENTS,  # total number of elements = M * N
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load scalar bias once per program
    x_fp32 = x.to(tl.float32)
    b_fp32 = tl.load(bias_ptr).to(tl.float32)

    # z = (x + b) * inv_div
    z = (x_fp32 + b_fp32) * inv_div

    # Swish: z * sigmoid(z)
    s = 1.0 / (1.0 + tl.exp(-z))
    y = (z * s).to(x.dtype)

    tl.store(y_ptr + offsets, y, mask=mask)
