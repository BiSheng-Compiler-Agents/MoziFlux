import triton
import triton.language as tl


@triton.jit
def _fused_tanh_scale_bias_maxpool2d(
    x_ptr,  # *f32 [B, C, H, W]
    bias_ptr,  # *f32 [C]
    y_ptr,  # *f32 [B, C, Hpo, Wpo]
    B,
    C,
    H,
    W,  # input dims
    HPO,
    WPO,  # pooled output dims
    STRIDE_B,
    STRIDE_C,
    STRIDE_H,
    STRIDE_W,  # input strides (in elements)
    O_STRIDE_B,
    O_STRIDE_C,
    O_STRIDE_H,
    O_STRIDE_W,  # output strides (in elements)
    scale,  # float scaling factor
    POOL_K: tl.
    constexpr,  # pooling kernel size (assume stride=POOL_K, padding=0, ceil_mode=False)
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    b = pid_bc // C
    c = pid_bc % C

    # Tiled coordinates in pooled space
    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]

    mask_hw = (oh < HPO) & (ow < WPO)

    # Base pointers for current (b, c) plane
    x_base = x_ptr + b * STRIDE_B + c * STRIDE_C
    y_base = y_ptr + b * O_STRIDE_B + c * O_STRIDE_C

    # Accumulator for max-pooling
    acc = tl.full((BLOCK_H, BLOCK_W), -float("inf"), tl.float32)

    # Precompute start indices in input feature map for each pooled output position
    ih0 = oh * POOL_K
    iw0 = ow * POOL_K
    ih0s = ih0 * STRIDE_H
    iw0s = iw0 * STRIDE_W

    # Iterate over POOL_K x POOL_K window
    for kh in range(POOL_K):
        ih_off = ih0s + kh * STRIDE_H
        for kw in range(POOL_K):
            ptrs = x_base + ih_off + iw0s + kw * STRIDE_W
            vals = tl.load(ptrs, mask=mask_hw, other=0.0).to(tl.float32)

            # Numerically stable tanh:
            # tanh(x) = sign(x) * (1 - e)/(1 + e), where e = exp(-2*|x|)
            abs_x = tl.abs(vals)
            e = tl.exp(-2.0 * abs_x)
            t = (1.0 - e) / (1.0 + e)
            sign = tl.where(vals >= 0, 1.0, -1.0)
            tanh_x = t * sign

            # Scale then max-reduce for pooling
            v = tanh_x * scale
            acc = tl.maximum(acc, v)

    # Add per-channel bias after max-pooling (equivalent since bias is constant per channel)
    bias_val = tl.load(bias_ptr + c).to(tl.float32)
    acc = acc + bias_val

    # Store results
    out_ptrs = y_base + oh * O_STRIDE_H + ow * O_STRIDE_W
    tl.store(out_ptrs, acc, mask=mask_hw)
