import triton
import triton.language as tl


@triton.jit
def _rmsnorm_nchw_kernel(
    x_ptr,
    y_ptr,
    rows,
    cols,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    eps,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    row_x_ptr = x_ptr + pid * stride_xm
    row_y_ptr = y_ptr + pid * stride_ym
    offsets = tl.arange(0, BLOCK_C)
    col_offs_x = offsets * stride_xn
    col_offs_y = offsets * stride_yn

    sumsq = tl.zeros([1], dtype=tl.float32)
    c = 0
    while c < cols:
        col_ids = c + offsets
        mask = col_ids < cols
        x = tl.load(row_x_ptr + c * stride_xn + col_offs_x,
                    mask=mask,
                    other=0.0)
        x_f32 = x.to(tl.float32)
        sumsq += tl.sum(x_f32 * x_f32, axis=0)
        c += BLOCK_C

    inv_rms = tl.rsqrt(sumsq / cols + eps)

    c = 0
    while c < cols:
        col_ids = c + offsets
        mask = col_ids < cols
        x = tl.load(row_x_ptr + c * stride_xn + col_offs_x,
                    mask=mask,
                    other=0.0)
        y = x * inv_rms
        tl.store(row_y_ptr + c * stride_yn + col_offs_y, y, mask=mask)
        c += BLOCK_C
