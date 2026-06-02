import triton
import triton.language as tl

@triton.jit
def _fused_bias_scale_clamp_inplace(
    in_out_ptr,     # *float*, tensor after conv_transpose, will be updated in-place
    bias_ptr,       # *float*, bias of shape [C]
    s,              # *float*, scaling factor
    n_elements,     # total number of elements = N * C * H * W
    C,              # number of channels
    HW,             # H * W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + arange
    mask = offsets < n_elements

    # Stream the large tensor via L2 to keep L1 available for the tiny bias array
    x = tl.load(in_out_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")

    # With BLOCK_SIZE == HW (set by the launcher), each program handles one (n, c) plane.
    plane_idx = block_start // HW
    ch = plane_idx % C
    b_scalar = tl.load(bias_ptr + ch)
    y = x + b_scalar

    # clamp to [0, 1], scale, clamp again, then divide by scale
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)
    y = y * s
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)
    y = y / s

    tl.store(in_out_ptr + offsets, y, mask=mask)
