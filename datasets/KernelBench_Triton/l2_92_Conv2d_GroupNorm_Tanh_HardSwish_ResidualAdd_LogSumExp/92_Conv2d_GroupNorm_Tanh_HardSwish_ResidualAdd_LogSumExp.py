import triton
import triton.language as tl


@triton.jit
def _fused_tanh_hswish_residual_lse(
    x_conv_ptr,
    x_norm_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    HW = H * W
    total = N * HW

    mask_pid = pid < total

    n = pid // HW
    q = pid % HW
    base = n * C * HW + q

    arange_c = tl.arange(0, BLOCK_C)

    m = -1.0e30
    c0 = 0
    while c0 < C:
        idx = c0 + arange_c
        ch_mask = (idx < C) & mask_pid
        offs = base + idx * HW

        xc = tl.load(x_conv_ptr + offs, mask=ch_mask,
                     other=-1.0e30).to(tl.float32)
        xn = tl.load(x_norm_ptr + offs, mask=ch_mask, other=0.0).to(tl.float32)

        t = 2.0 / (1.0 + tl.exp(-2.0 * xn)) - 1.0
        relu6 = tl.minimum(t + 3.0, 6.0)
        relu6 = tl.maximum(relu6, 0.0)
        hsw = t * (relu6 * (1.0 / 6.0))

        y = xc + hsw
        tile_max = tl.max(y, axis=0)
        m = tl.maximum(m, tile_max)
        c0 += BLOCK_C

    sum_exp = 0.0
    c0 = 0
    while c0 < C:
        idx = c0 + arange_c
        ch_mask = (idx < C) & mask_pid
        offs = base + idx * HW

        xc = tl.load(x_conv_ptr + offs, mask=ch_mask, other=0.0).to(tl.float32)
        xn = tl.load(x_norm_ptr + offs, mask=ch_mask, other=0.0).to(tl.float32)

        t = 2.0 / (1.0 + tl.exp(-2.0 * xn)) - 1.0
        relu6 = tl.minimum(t + 3.0, 6.0)
        relu6 = tl.maximum(relu6, 0.0)
        hsw = t * (relu6 * (1.0 / 6.0))

        y = xc + hsw
        e = tl.exp(y - m)
        e = tl.where(ch_mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)
        c0 += BLOCK_C

    lse = tl.log(sum_exp) + m
    tl.store(out_ptr + pid, lse, mask=mask_pid)
