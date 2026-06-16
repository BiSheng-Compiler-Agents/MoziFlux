import triton
import triton.language as tl


@triton.jit
def _softmax_sigmoid_fused_5d(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    stride_n,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * D * H * W
    row_mask = pid < total_rows

    w_idx = pid % W
    tmp = pid // W
    h_idx = tmp % H
    tmp = tmp // H
    d_idx = tmp % D
    n_idx = tmp // D

    base = (n_idx * stride_n + d_idx * stride_d + h_idx * stride_h +
            w_idx * stride_w).to(tl.int64)
    ch_offsets = tl.arange(0, BLOCK_C)

    m = -float("inf")
    c0 = 0
    while c0 < C:
        ch = c0 + ch_offsets
        ch_mask = ch < C
        ptrs = x_ptr + base + (ch * stride_c)
        x = tl.load(ptrs, mask=row_mask & ch_mask, other=-float("inf"))
        m = tl.maximum(m, tl.max(x.to(tl.float32), axis=0))
        c0 += BLOCK_C

    l = 0.0
    c0 = 0
    while c0 < C:
        ch = c0 + ch_offsets
        ch_mask = ch < C
        ptrs = x_ptr + base + (ch * stride_c)
        x = tl.load(ptrs, mask=row_mask & ch_mask,
                    other=-float("inf")).to(tl.float32)
        l += tl.sum(tl.exp(x - m), axis=0)
        c0 += BLOCK_C

    inv_l = 1.0 / l

    c0 = 0
    while c0 < C:
        ch = c0 + ch_offsets
        ch_mask = ch < C
        ptrs = x_ptr + base + (ch * stride_c)
        x = tl.load(ptrs, mask=row_mask & ch_mask,
                    other=-float("inf")).to(tl.float32)
        soft = tl.exp(x - m) * inv_l
        sig = 1.0 / (1.0 + tl.exp(-soft))
        out_ptrs = y_ptr + base + (ch * stride_c)
        tl.store(out_ptrs, sig, mask=row_mask & ch_mask)
        c0 += BLOCK_C
