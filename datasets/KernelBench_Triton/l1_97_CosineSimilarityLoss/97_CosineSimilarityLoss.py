import triton
import triton.language as tl

@triton.jit
def _cosine_similarity_rows_kernel(
    x_ptr, y_ptr, out_ptr,
    B, D,
    stride_xb, stride_xd,
    stride_yb, stride_yd,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_mask = pid < B

    offs = tl.arange(0, BLOCK_SIZE)
    mask = (offs < D) & row_mask

    x_row_ptr = x_ptr + pid * stride_xb
    y_row_ptr = y_ptr + pid * stride_yb

    x = tl.load(x_row_ptr + offs * stride_xd, mask=mask, other=0.0)
    y = tl.load(y_row_ptr + offs * stride_yd, mask=mask, other=0.0)

    x32 = x.to(tl.float32)
    y32 = y.to(tl.float32)

    dot = tl.sum(x32 * y32, axis=0)
    nx2 = tl.sum(x32 * x32, axis=0)
    ny2 = tl.sum(y32 * y32, axis=0)

    denom = tl.sqrt(nx2) * tl.sqrt(ny2)
    denom = tl.maximum(denom, EPS)
    loss = 1.0 - (dot / denom)

    tl.store(out_ptr + pid, loss, mask=row_mask)
