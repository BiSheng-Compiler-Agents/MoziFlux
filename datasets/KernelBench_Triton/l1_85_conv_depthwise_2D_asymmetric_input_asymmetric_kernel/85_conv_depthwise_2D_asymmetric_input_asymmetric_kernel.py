import triton
import triton.language as tl

@triton.jit
def dwconv2d_fwd_kernel(
    x_ptr,         # *fptr: [N, C, H, W] contiguous
    w_ptr,         # *fptr: [C, K_H*K_W] flattened contiguous
    b_ptr,         # *fptr: [C] or dummy (unused if BIAS=0)
    y_ptr,         # *fptr: [N, C, H_OUT, W_OUT] contiguous
    N, C, H, W,    # int32
    H_OUT, W_OUT,  # int32
    BIAS: tl.constexpr,     # 0/1
    K_H: tl.constexpr, K_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # program ids (over N*C and tiles of H_OUT*W_OUT) - do not change
    pid_nc = tl.program_id(0)
    pid_tile = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offs = pid_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    total_hw = H_OUT * W_OUT
    mask_o = offs < total_hw

    oh = offs // W_OUT
    ow = offs % W_OUT

    # Base pointers
    base_x = (n * C + c) * H * W
    base_y = (n * C + c) * H_OUT * W_OUT

    # Precompute origins for input coordinates
    oh_base = oh * STRIDE_H - PAD_H
    ow_base = ow * STRIDE_W - PAD_W

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Per-channel weight base
    w_ch_base = c * (K_H * K_W)

    # Ascend Triton does not accept chained boolean operators in a single if.
    if PAD_H == 0:
        if PAD_W == 0:
            if DIL_H == 1:
                if DIL_W == 1:
                    for kh in tl.static_range(K_H):
                        ih = oh_base + kh
                        row_ptrs = x_ptr + base_x + ih * W + ow_base
                        w_row_base = w_ptr + w_ch_base + kh * K_W
                        for kw in tl.static_range(K_W):
                            x_vals = tl.load(row_ptrs + kw, mask=mask_o, other=0.0)
                            w_val = tl.load(w_row_base + kw)
                            acc += x_vals.to(tl.float32) * w_val.to(tl.float32)
                else:
                    for kh in tl.static_range(K_H):
                        ih = oh_base + kh * DIL_H
                        h_ok = (ih >= 0) & (ih < H)
                        row_ptrs = x_ptr + base_x + ih * W + ow_base
                        w_row_base = w_ptr + w_ch_base + kh * K_W
                        for kw in tl.static_range(K_W):
                            iw = ow_base + kw * DIL_W
                            w_ok = (iw >= 0) & (iw < W)
                            m = mask_o & h_ok & w_ok
                            x_vals = tl.load(row_ptrs + kw * DIL_W, mask=m, other=0.0)
                            w_val = tl.load(w_row_base + kw)
                            acc += x_vals.to(tl.float32) * w_val.to(tl.float32)
            else:
                for kh in tl.static_range(K_H):
                    ih = oh_base + kh * DIL_H
                    h_ok = (ih >= 0) & (ih < H)
                    row_ptrs = x_ptr + base_x + ih * W + ow_base
                    w_row_base = w_ptr + w_ch_base + kh * K_W
                    for kw in tl.static_range(K_W):
                        iw = ow_base + kw * DIL_W
                        w_ok = (iw >= 0) & (iw < W)
                        m = mask_o & h_ok & w_ok
                        x_vals = tl.load(row_ptrs + kw * DIL_W, mask=m, other=0.0)
                        w_val = tl.load(w_row_base + kw)
                        acc += x_vals.to(tl.float32) * w_val.to(tl.float32)
        else:
            for kh in tl.static_range(K_H):
                ih = oh_base + kh * DIL_H
                h_ok = (ih >= 0) & (ih < H)
                row_ptrs = x_ptr + base_x + ih * W + ow_base
                w_row_base = w_ptr + w_ch_base + kh * K_W
                for kw in tl.static_range(K_W):
                    iw = ow_base + kw * DIL_W
                    w_ok = (iw >= 0) & (iw < W)
                    m = mask_o & h_ok & w_ok
                    x_vals = tl.load(row_ptrs + kw * DIL_W, mask=m, other=0.0)
                    w_val = tl.load(w_row_base + kw)
                    acc += x_vals.to(tl.float32) * w_val.to(tl.float32)
    else:
        # Generic path with full per-tap bounds checks.
        for kh in tl.static_range(K_H):
            ih = oh_base + kh * DIL_H
            h_ok = (ih >= 0) & (ih < H)
            row_ptrs = x_ptr + base_x + ih * W + ow_base
            w_row_base = w_ptr + w_ch_base + kh * K_W
            for kw in tl.static_range(K_W):
                iw = ow_base + kw * DIL_W
                w_ok = (iw >= 0) & (iw < W)
                m = mask_o & h_ok & w_ok
                x_vals = tl.load(row_ptrs + kw * DIL_W, mask=m, other=0.0)
                w_val = tl.load(w_row_base + kw)
                acc += x_vals.to(tl.float32) * w_val.to(tl.float32)

    if BIAS:
        b = tl.load(b_ptr + c)
        acc += b.to(tl.float32)

    tl.store(y_ptr + base_y + offs, acc, mask=mask_o)
