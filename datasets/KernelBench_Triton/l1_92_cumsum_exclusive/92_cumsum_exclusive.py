import triton
import triton.language as tl


@triton.jit
def _exclusive_cumsum_row_to_padded_kernel(
    x_ptr,
    y_ptr,
    rows,
    n_cols_in,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    BLOCK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row
    tl.store(row_y_ptr, 0.0)

    running = tl.zeros((), dtype=tl.float32)

    for chunk_idx in tl.static_range(NUM_CHUNKS):
        col_start = chunk_idx * BLOCK_SIZE
        base_x = row_x_ptr + col_start * stride_x_col
        base_y = row_y_ptr + (col_start + 1) * stride_y_col

        px = base_x
        py = base_y
        for offset in tl.static_range(BLOCK_SIZE):
            col = col_start + offset
            in_bounds = col < n_cols_in
            value = tl.load(px, mask=in_bounds, other=0.0).to(tl.float32)
            running += value
            tl.store(py, running, mask=in_bounds)
            px += stride_x_col
            py += stride_y_col
