import triton
import triton.language as tl

@triton.jit
def _fused_scale_maxpool3d_gap_clamp(
    x_ptr,                       # *x_dtype [N, C, D, H, W] contiguous
    out_ptr,                     # *x_dtype [N*C] flattened output
    N, C, D, H, W,               # dimensions
    scale,                       # scalar
    clamp_min, clamp_max,        # scalars
    KSIZE: tl.constexpr,         # pooling kernel size (assumed stride=KSIZE, padding=0, dilation=1)
    DP: tl.constexpr,            # pooled D
    HP: tl.constexpr,            # pooled H
    WP: tl.constexpr,            # pooled W
    NWINS: tl.constexpr,         # total pooling windows = DP * HP * WP
    BLOCK_WINS: tl.constexpr,    # number of windows processed per iteration
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    # Strides for a contiguous [N, C, D, H, W]
    sN = C * D * H * W
    sC = D * H * W
    sD = H * W
    sH = W
    sW = 1

    base = x_ptr + n * sN + c * sC

    offs = tl.arange(0, BLOCK_WINS)
    sumv = tl.zeros((), dtype=tl.float32)

    hpwp = HP * WP

    for start in range(0, NWINS, BLOCK_WINS):
        idx = start + offs
        mask = idx < NWINS

        dp = idx // hpwp
        rem = idx - dp * hpwp
        hp = rem // WP
        wp = rem - hp * WP

        di0 = dp * KSIZE
        hi0 = hp * KSIZE
        wi0 = wp * KSIZE

        base_offsets = di0 * sD + hi0 * sH + wi0 * sW

        # Initialize maxima as -inf in fp32
        maxi = tl.full(offs.shape, -float("inf"), tl.float32)

        # Iterate KSIZE^3 inside the window
        for kd in tl.static_range(0, KSIZE):
            for kh in tl.static_range(0, KSIZE):
                for kw in tl.static_range(0, KSIZE):
                    ptrs = base + base_offsets + kd * sD + kh * sH + kw * sW
                    v = tl.load(ptrs, mask=mask, other=-float("inf"))
                    vf32 = v.to(tl.float32) * scale
                    # For masked lanes, set to -inf so they don't affect max
                    vf32 = tl.where(mask, vf32, -float("inf"))
                    maxi = tl.maximum(maxi, vf32)

        # Sum valid maxima of this chunk
        maxi = tl.where(mask, maxi, 0.0)
        part = tl.sum(maxi, axis=0)
        sumv += part

    denom = tl.full((), NWINS, tl.float32)
    meanv = sumv / denom
    meanv = tl.minimum(tl.maximum(meanv, clamp_min), clamp_max)

    out_off = n * C + c
    tl.store(out_ptr + out_off, meanv)
