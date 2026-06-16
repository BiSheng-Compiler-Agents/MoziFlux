import triton
import triton.language as tl


@triton.jit
def _max_reduce_dim1_kernel(x_ptr, o_ptr, B, M, N, sx0, sx1, sx2, so0, so1,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Grid: (B, ceil_div(N, BLOCK_N))
    b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Hints for vectorization along N
    tl.multiple_of(n_offsets, 16)
    tl.max_contiguous(n_offsets, BLOCK_N)

    base = x_ptr + b * sx0

    # Initialize accumulator without touching output memory.
    o_ptrs = o_ptr + b * so0 + n_offsets * so1
    acc = tl.full([BLOCK_N], -float("inf"), tl.float32)

    # Stream across M rows, unrolled by BLOCK_M, and update running max
    m_start = 0
    while m_start < M:
        for mi in tl.static_range(0, BLOCK_M):
            m_idx = m_start + mi
            valid_row = m_idx < M
            row_ptrs = base + m_idx * sx1 + n_offsets * sx2
            row = tl.load(row_ptrs,
                          mask=n_mask & valid_row,
                          other=-float("inf"),
                          cache_modifier=".cg")
            acc = tl.maximum(acc, row)
        m_start += BLOCK_M

    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def _max_reduce_dim0_kernel(x_ptr, o_ptr, B, M, N, sx0, sx1, sx2, so0, so1,
                            BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr):
    # Grid: (M, ceil_div(N, BLOCK_N))
    m = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    o_ptrs = o_ptr + m * so0 + n_offsets * so1
    acc = tl.full([BLOCK_N], -float("inf"), tl.float32)

    b_start = 0
    while b_start < B:
        b_offsets = b_start + tl.arange(0, BLOCK_B)
        b_mask = b_offsets < B
        ptrs = x_ptr + b_offsets[:, None] * sx0 + m * sx1 + n_offsets[
            None, :] * sx2
        mask = b_mask[:, None] & n_mask[None, :]
        x_tile = tl.load(ptrs, mask=mask, other=-float("inf"))
        tile_max = tl.max(x_tile, axis=0).to(tl.float32)
        acc = tl.maximum(acc, tile_max)
        b_start += BLOCK_B

    tl.store(o_ptrs, acc, mask=n_mask)


@triton.jit
def _max_reduce_dim2_kernel(x_ptr, o_ptr, B, M, N, sx0, sx1, sx2, so0, so1,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Grid: (B, ceil_div(M, BLOCK_M))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # Single-buffered streaming across N
    n_start = 0
    # Initialize accumulator without touching output memory.
    o_ptrs = o_ptr + b * so0 + m_offsets * so1
    acc = tl.full([BLOCK_M], -float("inf"), tl.float32)

    while n_start < N:
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        ptrs = x_ptr + b * sx0 + m_offsets[:, None] * sx1 + n_offsets[
            None, :] * sx2
        mask = m_mask[:, None] & n_mask[None, :]
        x_tile = tl.load(ptrs,
                         mask=mask,
                         other=-float("inf"),
                         cache_modifier=".cg")

        tile_max = tl.max(x_tile, axis=1).to(tl.float32)
        acc = tl.maximum(acc, tile_max)
        n_start += BLOCK_N

    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=m_mask)
