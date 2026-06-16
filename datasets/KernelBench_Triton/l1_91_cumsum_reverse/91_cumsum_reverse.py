import triton
import triton.language as tl


@triton.jit
def _rcumsum_lastdim_kernel(
    x_ptr,
    y_ptr,
    rows,
    N,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per row
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    # Row base pointers
    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    # Lane indices within a tile
    i = tl.arange(0, BLOCK_N)

    # Accumulator that propagates across tiles from right to left
    carry = tl.zeros((), dtype=tl.float32)

    # Prefetch the rightmost tile
    last_block = NUM_BLOCKS - 1
    base = last_block * BLOCK_N
    rev_cols = base + (BLOCK_N - 1 - i)
    m_rev = rev_cols < N
    x_rev = tl.load(row_x_ptr + rev_cols * stride_x_col, mask=m_rev,
                    other=0.0).to(tl.float32)

    # Process tiles from rightmost to leftmost with simple software pipelining
    for b in range(NUM_BLOCKS):
        # Compute for the prefetched tile
        block_idx = NUM_BLOCKS - 1 - b
        base_cur = block_idx * BLOCK_N
        rev_cols_cur = base_cur + (BLOCK_N - 1 - i)
        m_rev_cur = rev_cols_cur < N

        # Inclusive scan within the reversed tile -> reverse-cumsum on original
        scan_rev = tl.cumsum(x_rev, axis=0)
        out_rev = scan_rev + carry
        tl.store(row_y_ptr + rev_cols_cur * stride_y_col,
                 out_rev.to(tl.float32),
                 mask=m_rev_cur)

        # Update carry with sum of this tile
        carry += tl.sum(x_rev, axis=0)

        # Prefetch next tile if any
        next_block_idx = block_idx - 1
        if next_block_idx >= 0:
            base_next = next_block_idx * BLOCK_N
            rev_cols_next = base_next + (BLOCK_N - 1 - i)
            m_rev_next = rev_cols_next < N
            x_rev = tl.load(row_x_ptr + rev_cols_next * stride_x_col,
                            mask=m_rev_next,
                            other=0.0).to(tl.float32)
