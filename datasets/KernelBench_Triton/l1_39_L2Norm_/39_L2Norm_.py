import triton
import triton.language as tl


@triton.jit
def _l2norm_rowwise_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Base pointers for this row
    row_x_ptr = x_ptr + pid * stride_xm
    row_y_ptr = y_ptr + pid * stride_ym

    cols = tl.arange(0, BLOCK_N)
    col_offs_x = cols * stride_xn
    col_offs_y = cols * stride_yn

    # First pass: compute L2 norm of the row in fp32
    sumsq = tl.zeros([1], dtype=tl.float32)
    n = 0
    while n < N:
        offs = n + cols
        mask = offs < N
        x = tl.load(row_x_ptr + (n * stride_xn) + col_offs_x,
                    mask=mask,
                    other=0.0)
        xf = x.to(tl.float32)
        sumsq += tl.sum(xf * xf, axis=0)
        n += BLOCK_N

    # Use reciprocal sqrt to reduce div latency; 0 -> inf, which yields NaN for 0*inf as in PyTorch (0/0)
    inv_norm = tl.rsqrt(sumsq)

    # Second pass: write normalized values
    n = 0
    while n < N:
        offs = n + cols
        mask = offs < N
        x = tl.load(row_x_ptr + (n * stride_xn) + col_offs_x,
                    mask=mask,
                    other=0.0)
        y = x * inv_norm
        tl.store(row_y_ptr + (n * stride_yn) + col_offs_y, y, mask=mask)
        n += BLOCK_N
