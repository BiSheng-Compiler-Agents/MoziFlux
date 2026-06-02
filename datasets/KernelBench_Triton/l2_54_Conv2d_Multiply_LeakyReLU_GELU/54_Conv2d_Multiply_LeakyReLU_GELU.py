import triton
import triton.language as tl

@triton.jit
def _fused_scale_lrelu_gelu(
    x_ptr,          # *float32, input tensor (NCHW) contiguous
    m_ptr,          # *float32, multiplier tensor flattened with shape (C,)
    y_ptr,          # *float32, output tensor (same shape as x)
    n_elements,     # int32, total elements B*C*H*W
    C,              # int32, number of channels
    HW,             # int32, product H*W
    negative_slope: tl.constexpr,  # float constant
    BLOCK_SIZE: tl.constexpr,      # tile size
):
    pid = tl.program_id(axis=0)
    arange = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + arange
    mask = offsets < n_elements
    tl.multiple_of(offsets, 16)

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")

    # Compute channel index for each element: ((idx // (H*W)) % C)
    plane_idx = offsets // HW
    c_idx = plane_idx % C
    scale = tl.load(m_ptr + c_idx, mask=mask, other=1.0)

    # Compute in fp32 for numerical stability and correctness
    x32 = x.to(tl.float32)
    s32 = scale.to(tl.float32)
    v = x32 * s32

    # Branchless LeakyReLU: v = v + (neg - 1) * min(v, 0)
    v = v + (negative_slope - 1.0) * tl.minimum(v, 0.0)

    # GELU (exact): 0.5 * v * (1 + erf(v / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    e = tl.erf(v * inv_sqrt2)
    y32 = 0.5 * v * (1.0 + e)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)
