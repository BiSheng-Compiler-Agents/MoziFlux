import triton
import triton.language as tl


@triton.jit
def _fused_leaky_mul_maxpool3d_2x2x2(
    x_ptr,  # *f32 [N, C, D, H, W]
    mult_ptr,  # *f32 [C, 1, 1, 1]
    y_ptr,  # *f32 [N, C, D//2, H//2, W//2]
    N,
    C,
    D,
    H,
    W,  # input sizes
    x_sN,
    x_sC,
    x_sD,
    x_sH,
    x_sW,  # x strides
    m_sC,  # multiplier stride along C dim
    oD,
    oH,
    oW,  # output sizes
    y_sN,
    y_sC,
    y_sD,
    y_sH,
    y_sW,  # y strides
    w_tiles,  # number of tiles along W for grid axis-2 decomposition
    NEG_SLOPE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program ids
    pid_nc = tl.program_id(0)  # ranges over N*C
    pid_d = tl.program_id(1)  # ranges over outD
    pid_hw = tl.program_id(2)  # ranges over outH * w_tiles

    # Decode (n, c)
    n = pid_nc // C
    c = pid_nc - n * C

    # Decode (h tile, w tile)
    ho = pid_hw // w_tiles
    wt = pid_hw - ho * w_tiles

    # Vector of output w indices this program instance computes
    wo = wt * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_wo = wo < oW

    # Corresponding input base indices (2x downsample window)
    d0 = 2 * pid_d
    h0 = 2 * ho
    w0_2 = 2 * wo

    # Base offsets (scalar base + vectorized W offsets)
    base_nc = n * x_sN + c * x_sC
    base_d0_h0 = base_nc + d0 * x_sD + h0 * x_sH
    base_w = base_d0_h0 + w0_2 * x_sW

    # Per-channel multiplier (broadcast across spatial dims)
    m = tl.load(mult_ptr + c * m_sC)

    # Shorthand strides
    sD = x_sD
    sH = x_sH
    sW = x_sW

    # 8 neighbors of the 2x2x2 pooling window (vectorized along W)
    o000 = base_w
    o001 = base_w + sW
    o010 = base_w + sH
    o011 = o010 + sW
    o100 = base_w + sD
    o101 = o100 + sW
    o110 = o100 + sH
    o111 = o110 + sW

    # Loads: only mask along Wout; D/H/W are guaranteed in-bounds for wo < oW
    v000 = tl.load(x_ptr + o000, mask=mask_wo, other=0.0)
    v001 = tl.load(x_ptr + o001, mask=mask_wo, other=0.0)
    v010 = tl.load(x_ptr + o010, mask=mask_wo, other=0.0)
    v011 = tl.load(x_ptr + o011, mask=mask_wo, other=0.0)
    v100 = tl.load(x_ptr + o100, mask=mask_wo, other=0.0)
    v101 = tl.load(x_ptr + o101, mask=mask_wo, other=0.0)
    v110 = tl.load(x_ptr + o110, mask=mask_wo, other=0.0)
    v111 = tl.load(x_ptr + o111, mask=mask_wo, other=0.0)

    # First LeakyReLU
    v000 = tl.maximum(v000, 0) + tl.minimum(v000, 0) * NEG_SLOPE
    v001 = tl.maximum(v001, 0) + tl.minimum(v001, 0) * NEG_SLOPE
    v010 = tl.maximum(v010, 0) + tl.minimum(v010, 0) * NEG_SLOPE
    v011 = tl.maximum(v011, 0) + tl.minimum(v011, 0) * NEG_SLOPE
    v100 = tl.maximum(v100, 0) + tl.minimum(v100, 0) * NEG_SLOPE
    v101 = tl.maximum(v101, 0) + tl.minimum(v101, 0) * NEG_SLOPE
    v110 = tl.maximum(v110, 0) + tl.minimum(v110, 0) * NEG_SLOPE
    v111 = tl.maximum(v111, 0) + tl.minimum(v111, 0) * NEG_SLOPE

    # Multiply by per-channel scalar
    v000 = v000 * m
    v001 = v001 * m
    v010 = v010 * m
    v011 = v011 * m
    v100 = v100 * m
    v101 = v101 * m
    v110 = v110 * m
    v111 = v111 * m

    # Second LeakyReLU
    v000 = tl.maximum(v000, 0) + tl.minimum(v000, 0) * NEG_SLOPE
    v001 = tl.maximum(v001, 0) + tl.minimum(v001, 0) * NEG_SLOPE
    v010 = tl.maximum(v010, 0) + tl.minimum(v010, 0) * NEG_SLOPE
    v011 = tl.maximum(v011, 0) + tl.minimum(v011, 0) * NEG_SLOPE
    v100 = tl.maximum(v100, 0) + tl.minimum(v100, 0) * NEG_SLOPE
    v101 = tl.maximum(v101, 0) + tl.minimum(v101, 0) * NEG_SLOPE
    v110 = tl.maximum(v110, 0) + tl.minimum(v110, 0) * NEG_SLOPE
    v111 = tl.maximum(v111, 0) + tl.minimum(v111, 0) * NEG_SLOPE

    # Max pooling over 2x2x2 window
    m0 = tl.maximum(v000, v001)
    m1 = tl.maximum(v010, v011)
    m2 = tl.maximum(v100, v101)
    m3 = tl.maximum(v110, v111)
    m4 = tl.maximum(m0, m1)
    m5 = tl.maximum(m2, m3)
    vout = tl.maximum(m4, m5)

    # Store result
    out_base = n * y_sN + c * y_sC + pid_d * y_sD + ho * y_sH
    tl.store(y_ptr + out_base + wo * y_sW, vout, mask=mask_wo)
