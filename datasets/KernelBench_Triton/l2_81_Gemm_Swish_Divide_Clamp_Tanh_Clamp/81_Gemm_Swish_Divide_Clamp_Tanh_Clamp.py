import triton
import triton.language as tl

@triton.jit
def _fused_epilogue_swish_div_clamp_tanh(
    x_ptr,  # input pointer
    y_ptr,  # output pointer
    n_elements,  # total number of elements
    BLOCK: tl.constexpr,  # block size
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # Load and upcast to fp32 for stable math
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # Swish: x * sigmoid(x) == x / (1 + exp(-x))
    y = xf / (1.0 + tl.exp(-xf))
    # Divide by 2
    y = y * 0.5

    # Clamp to [-1, 1]
    y = tl.maximum(y, -1.0)
    y = tl.minimum(y, 1.0)

    # tanh(y) using exp formulation: tanh(y) = (e^(2y) - 1) / (e^(2y) + 1)
    e2y = tl.exp(2.0 * y)
    y = (e2y - 1.0) / (e2y + 1.0)

    # Final clamp to [-1, 1]
    y = tl.maximum(y, -1.0)
    y = tl.minimum(y, 1.0)

    # Cast back and store
    y = y.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)
