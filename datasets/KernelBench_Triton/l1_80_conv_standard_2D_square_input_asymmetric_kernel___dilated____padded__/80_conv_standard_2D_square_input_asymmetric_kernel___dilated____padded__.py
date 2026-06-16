import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_fp32_kernel(
    x_ptr,  # (N, C, H, W) float32
    w_ptr,  # (K, OC) float32 where K = C*KH*KW (packed layout)
    b_ptr,  # (OC,) float32
    y_ptr,  # (N, OC, H_OUT, W_OUT) float32
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_p = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)

    # Offsets along pixels (flattened N*H_OUT*W_OUT) and out-channels
    p_offsets = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    total_p = N * H_OUT * W_OUT
    p_mask = p_offsets < total_p
    oc_mask = oc_offsets < OC

    # Decode flattened pixel indices into (n, y_out, x_out)
    WH_OUT = H_OUT * W_OUT
    n_idx = p_offsets // WH_OUT
    tmp = p_offsets % WH_OUT
    y_out = tmp // W_OUT
    x_out = tmp % W_OUT

    # Initialize accumulator with bias
    acc = tl.zeros((BLOCK_P, BLOCK_OC), dtype=tl.float32)
    bias = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Precompute constants to reduce index math
    HW = H * W
    KHW = KH * KW
    n_idx_C = n_idx * C
    n_idx_C_HW = n_idx_C * HW  # [BLOCK_P]

    # Iterate over kernel elements and channels
    for ky in range(KH):
        iy = y_out * STRIDE_H - PAD_H + ky * DIL_H  # [BLOCK_P]
        in_y_ok = (iy >= 0) & (iy < H)
        ky_base = ky * KW
        for kx in range(KW):
            ix = x_out * STRIDE_W - PAD_W + kx * DIL_W  # [BLOCK_P]
            in_x_ok = (ix >= 0) & (ix < W)
            mask_p = p_mask & in_y_ok & in_x_ok

            base_hw = iy * W + ix  # [BLOCK_P]
            x_base = n_idx_C_HW + base_hw  # [BLOCK_P]
            k_base_k = ky_base + kx  # in [0, KHW)

            for ci in range(C):
                x_index = x_base + ci * HW  # [BLOCK_P]
                x_vals = tl.load(x_ptr + x_index, mask=mask_p,
                                 other=0.0).to(tl.float32)

                k_id = ci * KHW + k_base_k
                w_index = k_id * OC + oc_offsets  # [BLOCK_OC]
                w_vals = tl.load(w_ptr + w_index, mask=oc_mask,
                                 other=0.0).to(tl.float32)

                acc += x_vals[:, None] * w_vals[None, :]

    # Store results to Y
    y_index = ((n_idx[:, None] * OC + oc_offsets[None, :]) * H_OUT +
               y_out[:, None]) * W_OUT + x_out[:, None]
    store_mask = p_mask[:, None] & oc_mask[None, :]
    tl.store(y_ptr + y_index, acc, mask=store_mask)
