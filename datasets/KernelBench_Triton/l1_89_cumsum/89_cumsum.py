import triton
import triton.language as tl

@triton.jit
def _rowwise_cumsum_kernel(
    x_ptr,
    y_ptr,
    carry_in_ptr,
    carry_out_ptr,
    N,
    chunk_start,
    stride_x0, stride_x1,
    stride_y0, stride_y1,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_base = x_ptr + row * stride_x0
    y_row_base = y_ptr + row * stride_y0
    carry = tl.load(carry_in_ptr + row)

    for offset in tl.static_range(0, BLOCK_N):
        col = chunk_start + offset
        mask = col < N
        value = tl.load(x_row_base + col * stride_x1, mask=mask, other=0.0)
        carry += value.to(tl.float32)
        tl.store(y_row_base + col * stride_y1, carry, mask=mask)

    tl.store(carry_out_ptr + row, carry)
