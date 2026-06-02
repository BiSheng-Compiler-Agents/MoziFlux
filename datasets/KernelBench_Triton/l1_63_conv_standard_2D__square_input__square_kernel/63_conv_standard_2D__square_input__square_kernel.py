import triton
import triton.language as tl

@triton.jit
def conv2d_nchw_s1p0_vecoc_kernel(
    x_ptr,        # *f32
    w_ptr,        # *f32
    b_ptr,        # *f32 or dummy
    y_ptr,        # *f32
    N,            # int32 runtime
    H,            # int32 runtime
    W,            # int32 runtime
    OC,           # int32 runtime
    H_out,        # int32 runtime
    W_out,        # int32 runtime
    TILES_WO,     # int32 runtime: number of tiles along width
    C: tl.constexpr,           # compile-time in_channels
    K: tl.constexpr,           # compile-time kernel_size (square)
    BIAS: tl.constexpr,        # 0/1 compile-time whether to add bias
    BLOCK_HO: tl.constexpr,    # tile size along output height
    BLOCK_WO: tl.constexpr,    # tile size along output width
    BLOCK_OC: tl.constexpr,    # number of output channels computed per program
):
    pid_n = tl.program_id(axis=0)
    pid_ob = tl.program_id(axis=1)  # output-channel block id
    pid_tile = tl.program_id(axis=2)

    # Decode spatial tile index
    tile_ho = pid_tile // TILES_WO
    tile_wo = pid_tile % TILES_WO

    ho_offsets = tile_ho * BLOCK_HO + tl.arange(0, BLOCK_HO)
    wo_offsets = tile_wo * BLOCK_WO + tl.arange(0, BLOCK_WO)
    OH = ho_offsets[:, None]  # [BH, 1]
    OW = wo_offsets[None, :]  # [1, BW]

    mask_spatial = (OH < H_out) & (OW < W_out)

    # Strides for NCHW contiguous layout
    x_n_stride = C * H * W
    x_c_stride = H * W
    x_h_stride = W
    x_w_stride = 1

    y_n_stride = OC * H_out * W_out
    y_oc_stride = H_out * W_out
    y_h_stride = W_out
    y_w_stride = 1

    # Output channels this program computes
    oc_offsets = pid_ob * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = oc_offsets < OC

    # Base pointers for this batch
    x_base = pid_n * x_n_stride
    y_base = pid_n * y_n_stride

    # Accumulator: [BLOCK_OC, BLOCK_HO, BLOCK_WO]
    acc = tl.zeros((BLOCK_OC, BLOCK_HO, BLOCK_WO), dtype=tl.float32)

    # Direct convolution: y[n, oc, oh, ow] = sum_{c, kh, kw} x[n, c, oh+kh, ow+kw] * w[oc, c, kh, kw]
    for kh in range(0, K):
        for kw in range(0, K):
            ih = OH + kh
            iw = OW + kw
            x_hw_offsets = x_base + ih * x_h_stride + iw * x_w_stride  # [BH, BW]
            # For stride=1, padding=0, dilation=1, these are always in-bounds, but keep mask for safety
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            for c in range(0, C):
                # Load weights for a vector of OC
                w_off = ((oc_offsets * C + c) * K + kh) * K + kw  # [BLOCK_OC]
                w_vec = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0).to(tl.float32)  # [BLOCK_OC]
                # Load input tile for this (c, kh, kw)
                x_offs_c = x_hw_offsets + c * x_c_stride  # [BH, BW]
                x_val = tl.load(x_ptr + x_offs_c, mask=mask_spatial & in_bounds, other=0.0).to(tl.float32)  # [BH, BW]
                # FMA with broadcasting over OC
                acc += w_vec[:, None, None] * x_val[None, :, :]

    if BIAS:
        b_vec = tl.load(b_ptr + oc_offsets, mask=mask_oc, other=0.0).to(tl.float32)  # [BLOCK_OC]
        acc += b_vec[:, None, None]

    # Store results
    y_offsets = (
        y_base
        + oc_offsets[:, None, None] * y_oc_stride
        + OH[None, :, :] * y_h_stride
        + OW[None, :, :] * y_w_stride
    )
    store_mask = mask_oc[:, None, None] & mask_spatial[None, :, :]
    tl.store(y_ptr + y_offsets, acc, mask=store_mask)
