import triton
import triton.language as tl


@triton.jit
def _pool_gelu_scale_max_kernel(
        x_ptr,  # *[B, F], row-major
        out_ptr,  # *[B]
        stride_x,  # stride between rows in elements
        G,  # number of pooling groups = F // K
        scale,  # scaling factor (float)
        K: tl.constexpr,  # pool kernel size
        BLOCK_G: tl.
    constexpr,  # number of groups processed per row (power-of-two >= G)
):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * stride_x

    # group offsets
    offs_g = tl.arange(0, BLOCK_G)  # [BLOCK_G]
    mask_g = offs_g < G  # [BLOCK_G]

    # base pointer for each group's start
    base = row_ptr + offs_g * K

    # compute pooled sums by streaming over K (specialize K==4)
    sums = tl.zeros((BLOCK_G, ), dtype=tl.float32)
    if K == 4:
        v0 = tl.load(base + 0, mask=mask_g, other=0.0)
        v1 = tl.load(base + 1, mask=mask_g, other=0.0)
        v2 = tl.load(base + 2, mask=mask_g, other=0.0)
        v3 = tl.load(base + 3, mask=mask_g, other=0.0)
        sums = v0.to(tl.float32) + v1.to(tl.float32) + v2.to(
            tl.float32) + v3.to(tl.float32)
    else:
        for kk in range(0, K):
            vals = tl.load(base + kk, mask=mask_g, other=0.0)
            sums += vals.to(tl.float32)

    # choose max or min via a single reduction using sign trick
    s = scale
    sign = tl.where(s >= 0, 1.0, -1.0)
    signed = sums * sign
    signed = tl.where(mask_g, signed, -float("inf"))
    best_signed = tl.max(signed, axis=0)
    sel_sum = best_signed * sign

    invK = 1.0 / K
    sel_mean = sel_sum * invK

    # exact GELU via erf on selected mean only
    inv_sqrt2 = 0.7071067811865475
    gelu_sel = 0.5 * sel_mean * (1.0 + tl.math.erf(sel_mean * inv_sqrt2))

    res = gelu_sel * s
    tl.store(out_ptr + pid, res)
