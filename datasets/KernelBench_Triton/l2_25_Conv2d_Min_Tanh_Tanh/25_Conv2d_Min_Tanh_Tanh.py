import triton
import triton.language as tl


@triton.jit
def _min_tanh2_nchw_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    stride_nx,
    stride_cx,
    stride_hx,
    stride_wx,
    stride_ny,
    stride_cy,
    stride_hy,
    stride_wy,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)

    hw_start = pid_hw * BLOCK_HW
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (H * W)

    # Compute h, w from flattened hw index
    h_idx = offs_hw // W
    w_idx = offs_hw - h_idx * W

    # Base pointers for the current N and (h,w) locations
    base_x = pid_n * stride_nx + h_idx * stride_hx + w_idx * stride_wx

    # Initialize min with the first channel to preserve dtype
    x0 = tl.load(x_ptr + base_x + 0 * stride_cx, mask=mask_hw, other=0.0)
    min_vals = x0

    # Fast path: small C (<=16) fully unrolled with masked loads to cut loop overhead
    # Use large positive "other" so masked channels don't affect minimum
    BIG_POS = 1e30
    if C <= 16:
        for k in tl.static_range(1, 16):
            vals = tl.load(x_ptr + base_x + k * stride_cx,
                           mask=mask_hw & (k < C),
                           other=BIG_POS)
            min_vals = tl.minimum(min_vals, vals)
    else:
        # Unroll channel reduction to improve ILP and reduce loop overhead
        c = 1
        while c + 3 < C:
            base_c = base_x + c * stride_cx
            v1 = tl.load(x_ptr + base_c + 0 * stride_cx,
                         mask=mask_hw,
                         other=0.0)
            v2 = tl.load(x_ptr + base_c + 1 * stride_cx,
                         mask=mask_hw,
                         other=0.0)
            v3 = tl.load(x_ptr + base_c + 2 * stride_cx,
                         mask=mask_hw,
                         other=0.0)
            v4 = tl.load(x_ptr + base_c + 3 * stride_cx,
                         mask=mask_hw,
                         other=0.0)
            pair12 = tl.minimum(v1, v2)
            pair34 = tl.minimum(v3, v4)
            block_min = tl.minimum(pair12, pair34)
            min_vals = tl.minimum(min_vals, block_min)
            c += 4

        while c < C:
            vals = tl.load(x_ptr + base_x + c * stride_cx,
                           mask=mask_hw,
                           other=0.0)
            min_vals = tl.minimum(min_vals, vals)
            c += 1

    t1 = tl.tanh(min_vals)
    t2 = tl.tanh(t1)

    base_y = pid_n * stride_ny + 0 * stride_cy + h_idx * stride_hy + w_idx * stride_wy
    tl.store(y_ptr + base_y, t2, mask=mask_hw)
