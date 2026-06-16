import triton
import triton.language as tl


@triton.jit
def _fused_maxpool2x2_hardtanh_mean_tanh(
    x_ptr,  # ptr to input [B, C, H, W]
    out_ptr,  # ptr to output [B, C, 1, 1]
    C,
    H,
    W,  # tensor sizes
    x_stride_b,
    x_stride_c,
    x_stride_h,
    x_stride_w,  # input strides
    o_stride_b,
    o_stride_c,  # output strides
    hard_min,
    hard_max,  # hardtanh bounds (float)
    H_OUT: tl.constexpr,  # pooled H = floor(H/2)
    W_OUT: tl.constexpr,  # pooled W = floor(W/2)
    BLOCK_W: tl.constexpr  # tile size covering H_OUT*W_OUT (power-of-two)
):
    # One program per (b, c)
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    # Base offsets
    x_base = b * x_stride_b + c * x_stride_c
    o_base = b * o_stride_b + c * o_stride_c

    # Total pooled elements (compile-time constant)
    TOT = H_OUT * W_OUT

    # Vectorize across all pooled positions in a single pass
    offs = tl.arange(0, BLOCK_W)
    mask = offs < TOT

    # Decode (oh, ow) from linear index
    oh = offs // W_OUT
    ow = offs - oh * W_OUT

    # 2x2 pooling window (stride=2, kernel=2)
    r0 = 2 * oh
    r1 = r0 + 1
    c0 = 2 * ow
    c1 = c0 + 1

    row0 = r0 * x_stride_h
    row1 = r1 * x_stride_h
    col0 = c0 * x_stride_w
    col1 = c1 * x_stride_w

    ptr00 = x_ptr + x_base + row0 + col0
    ptr01 = x_ptr + x_base + row0 + col1
    ptr10 = x_ptr + x_base + row1 + col0
    ptr11 = x_ptr + x_base + row1 + col1

    v00 = tl.load(ptr00, mask=mask, other=-float("inf"))
    v01 = tl.load(ptr01, mask=mask, other=-float("inf"))
    v10 = tl.load(ptr10, mask=mask, other=-float("inf"))
    v11 = tl.load(ptr11, mask=mask, other=-float("inf"))

    vmax = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    # hardtanh clamp
    vmax = tl.minimum(tl.maximum(vmax, hard_min), hard_max)

    # Accumulate mean over all pooled elements
    vmax = tl.where(mask, vmax, 0.0)
    acc = tl.sum(vmax, axis=0)
    mean_val = acc * (1.0 / (H_OUT * W_OUT))

    # tanh using stable exp formulation
    e2 = tl.exp(2.0 * mean_val)
    out_val = 1.0 - 2.0 / (e2 + 1.0)

    # Store to [B, C, 1, 1]
    tl.store(out_ptr + o_base, out_val)
