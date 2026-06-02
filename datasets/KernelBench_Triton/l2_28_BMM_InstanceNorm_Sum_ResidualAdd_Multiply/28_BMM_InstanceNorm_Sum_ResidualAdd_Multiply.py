import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=["F", "BLOCK"],
)
@triton.jit
def _rownorm_addmul_kernel(
    x_ptr,      # pointer to [B, F] input (after linear)
    y_ptr,      # pointer to [B, F] input y
    out_ptr,    # pointer to [B, F] output
    B,          # number of rows (batch size)
    F,          # number of features (out_features)
    stride_x,   # stride between consecutive rows of x in elements
    stride_y,   # stride between consecutive rows of y in elements
    stride_out, # stride between consecutive rows of out in elements
    eps,        # epsilon for numerical stability
    inv_F,      # 1.0 / F
    BLOCK: tl.constexpr,  # block size (next power of 2 >= F)
):
    pid = tl.program_id(0)  # row id
    offs = tl.arange(0, BLOCK)
    # Hints for better codegen on contiguous rows
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK)

    # Masks to guard OOB
    row_mask = pid < B
    col_mask = offs < F
    mask = row_mask & col_mask

    # Base pointers for this row
    x_row_ptr = x_ptr + pid * stride_x + offs
    y_row_ptr = y_ptr + pid * stride_y + offs
    out_row_ptr = out_ptr + pid * stride_out + offs

    # Load row slices
    x_row = tl.load(x_row_ptr, mask=mask, other=0.0)
    y_row = tl.load(y_row_ptr, mask=mask, other=0.0)

    # Compute mean and variance across the row directly from x_row
    sum_x = tl.sum(x_row, axis=0)
    sum_x2 = tl.sum(x_row * x_row, axis=0)
    mean = sum_x * inv_F
    var = sum_x2 * inv_F - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)

    # Fuse normalize + add + mul: (x_hat + y) * y
    # out = y*y + y*(x - mean)*rstd
    y_sq = y_row * y_row
    out_row = y_sq + y_row * (x_row - mean) * rstd

    tl.store(out_row_ptr, out_row, mask=mask)
