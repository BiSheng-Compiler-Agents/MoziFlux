import triton
import triton.language as tl


@triton.jit
def _bias_sub_tanh_kernel(
    x_ptr,  # *T: input/output tensor (N, C, H, W) flattened
    b_ptr,  # *T: bias tensor (C)
    y_ptr,  # *T: output tensor (same as x)
    HW: tl.constexpr,  # H * W
    C: tl.constexpr,  # number of channels
    NCHW: tl.constexpr,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NCHW

    # Compute channel index for each element in flattened NCHW layout
    c_idx = (offs // HW) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + c_idx, mask=mask, other=0.0)

    x32 = x.to(tl.float32)
    b32 = b.to(tl.float32)
    z = x32 - b32
    y = tl.tanh(z)

    # Store back; Triton will cast to the destination pointer dtype if needed
    tl.store(y_ptr + offs, y, mask=mask)
