import triton
import triton.language as tl

@triton.jit
def _softmax_row_fwd_db_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x_row_ptr = x_ptr + pid * stride_row
    y_row_ptr = y_ptr + pid * stride_row

    x = tl.load(x_row_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(x, axis=0)
    numerators = tl.exp(x - row_max)
    denom = tl.sum(numerators, axis=0)
    y = numerators / denom
    tl.store(y_row_ptr + offs, y, mask=mask)
