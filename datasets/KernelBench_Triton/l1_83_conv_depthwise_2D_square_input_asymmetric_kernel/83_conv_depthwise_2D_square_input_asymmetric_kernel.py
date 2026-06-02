import triton
import triton.language as tl

@triton.jit
def _dw_conv_kh1_kernel(
    x_ptr,             # *const T, [B, C, H_in, W_in]
    w_ptr,             # *const T, [C, 1, K, 1]
    b_ptr,             # *const T or nullptr, [C]
    y_ptr,             # *T, [B, C, H_out, W_out]
    B, C,
    H_in, W_in,
    H_out, W_out,
    K,
    S_h, S_w,
    P_h, P_w,
    D_h, D_w,
    x_bs, x_cs, x_hs, x_ws,
    w_cs, w_khs,
    y_bs, y_cs, y_hs, y_ws,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_bc // C
    c = pid_bc % C

    oh_ids = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow_ids = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_mask = (oh_ids[:, None] < H_out) & (ow_ids[None, :] < W_out)

    # Compute corresponding input column indices for kW=1 and validity
    iw0 = ow_ids * S_w - P_w
    valid_w = (iw0[None, :] >= 0) & (iw0[None, :] < W_in)

    # Base pointers for (b, c)
    x_bc_ptr = x_ptr + b * x_bs + c * x_cs
    y_bc_ptr = y_ptr + b * y_bs + c * y_cs

    # Precompute base offsets along width for the tile to reduce address arithmetic
    x_col_base = x_bc_ptr + iw0[None, :] * x_ws

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Double-buffered software pipelining across K dimension
    # Precompute base ih for kh=0
    oh_base = oh_ids * S_h - P_h

    # Handle K >= 1
    if K > 0:
        ih0 = oh_base + 0 * D_h
        valid_h0 = (ih0[:, None] >= 0) & (ih0[:, None] < H_in)
        mask0 = out_mask & valid_h0 & valid_w
        x_ptrs0 = x_col_base + ih0[:, None] * x_hs
        x_vals0 = tl.load(x_ptrs0, mask=mask0, other=0.0).to(tl.float32)
        w_base_ptr = w_ptr + c * w_cs
        w0 = tl.load(w_base_ptr + 0 * w_khs).to(tl.float32)

        # Main pipelined loop
        for kh in range(0, K - 1):
            khn = kh + 1
            ih1 = oh_base + khn * D_h
            valid_h1 = (ih1[:, None] >= 0) & (ih1[:, None] < H_in)
            mask1 = out_mask & valid_h1 & valid_w
            x_ptrs1 = x_col_base + ih1[:, None] * x_hs
            x_vals1 = tl.load(x_ptrs1, mask=mask1, other=0.0).to(tl.float32)
            w1 = tl.load(w_base_ptr + khn * w_khs).to(tl.float32)

            # FMA accumulate current buffer
            acc += x_vals0 * w0

            # Rotate buffers
            x_vals0 = x_vals1
            w0 = w1

        # Final buffered step
        acc += x_vals0 * w0

    if HAS_BIAS:
        b_val = tl.load(b_ptr + c).to(tl.float32)
        acc += b_val

    # Store results (cast handled by tl.store to y_ptr dtype)
    y_ptrs = y_bc_ptr + oh_ids[:, None] * y_hs + ow_ids[None, :] * y_ws
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)
