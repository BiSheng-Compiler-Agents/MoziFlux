import triton
import triton.language as tl

@triton.jit
def _cumsum_lastdim_kernel(
    x_ptr,      # *dtype
    out_ptr,    # *dtype
    M,          # int32
    N,          # int32
    stride_xm,  # int32
    stride_xn,  # int32
    stride_om,  # int32
    stride_on,  # int32
    BLOCK_N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    row_x = x_ptr + pid_m * stride_xm
    row_o = out_ptr + pid_m * stride_om

    offs = tl.arange(0, BLOCK_N)
    carry = tl.zeros((), dtype=tl.float32)

    for block_idx in range(NUM_BLOCKS):
        cols = block_idx * BLOCK_N + offs
        mask = cols < N
        vals = tl.load(row_x + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        scan = tl.cumsum(vals, axis=0) + carry
        tl.store(row_o + cols * stride_on, scan, mask=mask)
        carry += tl.sum(vals, axis=0)
