import triton
import triton.language as tl


@triton.jit
def _scale_min_channel_kernel(
    x_ptr,
    y_ptr,
    scale,
    B,
    H,
    W,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    C: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    hw_start = pid_hw * BLOCK_HW
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask_b = b_idx < B
    mask_hw = offs_hw < (H * W)
    h = offs_hw // W
    w = offs_hw % W

    x_base = (b_idx[:, None] * stride_xn + h[None, :] * stride_xh +
              w[None, :] * stride_xw)
    offs_c = tl.arange(0, BLOCK_C)
    acc = tl.full((BLOCK_B, BLOCK_HW), float("inf"), tl.float32)

    for c0 in range(0, C, BLOCK_C):
        c_idx = c0 + offs_c[None, None, :]
        mask_c = c_idx < C
        mask = mask_b[:, None, None] & mask_hw[None, :, None] & mask_c
        x_vals = tl.load(
            x_ptr + x_base[:, :, None] + c_idx * stride_xc,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scaled_vals = tl.where(mask, x_vals * scale, float("inf"))
        block_min = tl.min(scaled_vals, axis=2).to(tl.float32)
        acc = tl.minimum(acc, block_min)

    y_offs = (b_idx[:, None] * stride_yn + h[None, :] * stride_yh +
              w[None, :] * stride_yw)
    tl.store(
        y_ptr + y_offs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_b[:, None] & mask_hw[None, :],
    )
