import triton
import triton.language as tl

@triton.jit
def _fused_mish_add_hardtanh_scale_kernel(
    x_ptr, y_ptr,
    n_elements,
    add_value, scale_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hints for better vectorization/coalescing
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    mask = offs < n_elements

    # Load; values are single-use so prefer evict_last
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last")
    x_f32 = x.to(tl.float32)

    # Mish: x * tanh(softplus(x))
    # Stable softplus: max(x, 0) + log(1 + exp(-abs(x)))
    ax = tl.abs(x_f32)
    sp = tl.maximum(x_f32, 0.0) + tl.log(1.0 + tl.exp(-ax))

    # Stable tanh via exp avoids backend-specific libdevice calls.
    s2 = 2.0 * sp
    tanh_sp = 2.0 / (1.0 + tl.exp(-s2)) - 1.0
    mish = x_f32 * tanh_sp

    # Add, clamp to [-1, 1] (hardtanh), then scale
    out = mish + add_value
    out = tl.minimum(tl.maximum(out, -1.0), 1.0)
    out = (out * scale_value).to(x.dtype)

    tl.store(y_ptr + offs, out, mask=mask)
