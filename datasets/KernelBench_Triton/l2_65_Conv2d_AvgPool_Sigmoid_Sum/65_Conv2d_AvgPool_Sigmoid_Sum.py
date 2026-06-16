import triton
import triton.language as tl


@triton.jit
def _pool_sigmoid_channel_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    STRIDE_B,
    STRIDE_C,
    STRIDE_H,
    STRIDE_W,
    OUT_STRIDE_B,
    OUT_STRIDE_C,
    K: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    if b >= B:
        return

    H_OUT = H // K
    W_OUT = W // K
    inv_area = 1.0 / (K * K)

    base = x_ptr + b * STRIDE_B + c * STRIDE_C
    total = tl.zeros((BLOCK_W, ), dtype=tl.float32)
    lane = tl.arange(0, BLOCK_W)

    for h_out in range(0, H_OUT):
        h_base = h_out * K
        for w_start in range(0, W_OUT, BLOCK_W):
            w_out = w_start + lane
            mask = w_out < W_OUT
            acc = tl.zeros((BLOCK_W, ), dtype=tl.float32)
            for ky in tl.static_range(0, K):
                row = base + (h_base + ky) * STRIDE_H
                for kx in tl.static_range(0, K):
                    ptrs = row + (w_out * K + kx) * STRIDE_W
                    vals = tl.load(ptrs, mask=mask, other=0.0)
                    acc += vals
            avg = acc * inv_area
            sig = 1.0 / (1.0 + tl.exp(-avg))
            total += tl.where(mask, sig, 0.0)

    partial = tl.sum(total, axis=0)
    tl.store(out_ptr + b * OUT_STRIDE_B + c * OUT_STRIDE_C, partial)


@triton.jit
def _sum_channels_kernel(
    x_ptr,
    out_ptr,
    C,
    STRIDE_B,
    STRIDE_C,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    vals = tl.load(x_ptr + pid * STRIDE_B + offs * STRIDE_C,
                   mask=mask,
                   other=0.0)
    total = tl.sum(vals, axis=0)
    tl.store(out_ptr + pid, total)
