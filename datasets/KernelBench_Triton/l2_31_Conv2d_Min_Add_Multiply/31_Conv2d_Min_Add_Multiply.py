import triton
import triton.language as tl

@triton.jit
def _fused_min_bias_scale_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    hw,
    channels,
    const_value,
    scaling,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c_idx = (offs // hw) % channels
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    clipped = tl.minimum(x, const_value)
    out = (clipped + bias) * scaling
    tl.store(out_ptr + offs, out, mask=mask)
