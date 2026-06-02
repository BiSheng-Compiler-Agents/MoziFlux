import triton
import triton.language as tl

@triton.jit
def _relu_add_bias_kernel(
    x_ptr,
    y_ptr,
    b_ptr,
    N,
    C,
    H,
    W,
    BLOCK_W: tl.constexpr,
):
    pid_nch = tl.program_id(axis=0)
    pid_wblk = tl.program_id(axis=1)

    n = pid_nch // (C * H)
    rem = pid_nch % (C * H)
    c = rem // H
    h = rem % H

    start_w = pid_wblk * BLOCK_W
    w_offsets = start_w + tl.arange(0, BLOCK_W)
    mask = w_offsets < W

    base = ((n * C + c) * H + h) * W
    offs = base + w_offsets
    tl.max_contiguous(offs, BLOCK_W)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    b = tl.load(b_ptr + c)
    x = x + b
    tl.store(y_ptr + offs, x, mask=mask)
