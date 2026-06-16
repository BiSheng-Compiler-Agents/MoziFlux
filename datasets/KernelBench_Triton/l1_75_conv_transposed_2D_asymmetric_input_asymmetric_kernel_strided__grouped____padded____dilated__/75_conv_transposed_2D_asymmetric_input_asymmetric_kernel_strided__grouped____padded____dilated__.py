import triton
import triton.language as tl


@triton.jit
def _upsample_insert_zeros_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    H_UP,
    W_UP,
    STRIDE_H,
    STRIDE_W,
    in_strideN,
    in_strideC,
    in_strideH,
    in_strideW,
    out_strideN,
    out_strideC,
    out_strideH,
    out_strideW,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    hw_start = pid_hw * BLOCK_HW
    offs = hw_start + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)

    h_idx = offs // W
    w_idx = offs - h_idx * W

    x_base = x_ptr + n * in_strideN + c * in_strideC
    y_base = y_ptr + n * out_strideN + c * out_strideC

    vals = tl.load(x_base + h_idx * in_strideH + w_idx * in_strideW,
                   mask=mask,
                   other=0)

    ho = h_idx * STRIDE_H
    wo = w_idx * STRIDE_W

    tl.store(y_base + ho * out_strideH + wo * out_strideW, vals, mask=mask)
