import triton
import triton.language as tl


@triton.jit
def _reduce_dim1_kernel(
        x_ptr,
        out_ptr,
        B,
        M,
        N,  # reduce over dim=1 (M)
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr):
    # PIDs: (b, n-tile)
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Hints for vectorization on fully-aligned tiles
    tl.max_contiguous(n_offsets, BLOCK_N)

    # Accumulator across M for a vector of N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    base_b = pid_b * M * N

    # Stream rows to minimize register pressure: load one row at a time
    for k0 in range(0, tl.cdiv(M, BLOCK_K)):
        k_base = k0 * BLOCK_K
        # Unrolled inner loop across a small chunk of M
        for kk in tl.static_range(0, BLOCK_K):
            k_idx = k_base + kk
            row_ptrs = x_ptr + base_b + k_idx * N + n_offsets
            # Masked for tail rows or tail N
            vals = tl.load(row_ptrs, mask=mask_n & (k_idx < M), other=0.0)
            acc += vals.to(tl.float32)

    # store to out: shape [B, 1, N] contiguous -> offset b*N + n
    out_ptrs = out_ptr + pid_b * N + n_offsets
    tl.store(out_ptrs, acc, mask=mask_n)


@triton.jit
def _reduce_dim2_kernel(
        x_ptr,
        out_ptr,
        B,
        M,
        N,  # reduce over dim=2 (N)
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m_tile = tl.program_id(1)

    m_start = pid_m_tile * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    base_b = pid_b * M * N
    # Stream over contiguous N in tiles for coalesced loads
    for k in range(0, tl.cdiv(N, BLOCK_K)):
        k_offsets = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < N
        ptrs = x_ptr + base_b + m_offsets[:, None] * N + k_offsets[None, :]
        # Use maskless fast path when tile fully inside bounds
        if (m_start + BLOCK_M <= M) and (k * BLOCK_K + BLOCK_K <= N):
            x = tl.load(ptrs)
        else:
            x = tl.load(ptrs,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0)
        x = x.to(tl.float32)
        acc += tl.sum(x, axis=1)

    # store to out: shape [B, M, 1] contiguous -> offset b*M + m
    out_ptrs = out_ptr + pid_b * M + m_offsets
    tl.store(out_ptrs, acc, mask=mask_m)


@triton.jit
def _reduce_dim0_kernel(
        x_ptr,
        out_ptr,
        B,
        M,
        N,  # reduce over dim=0 (B)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_KB: tl.constexpr):
    pid_m_tile = tl.program_id(0)
    pid_n_tile = tl.program_id(1)

    m_start = pid_m_tile * BLOCK_M
    n_start = pid_n_tile * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # loop over B in chunks (streaming to keep register usage low)
    for kb in range(0, tl.cdiv(B, BLOCK_KB)):
        b_offsets = kb * BLOCK_KB + tl.arange(0, BLOCK_KB)
        mask_b = b_offsets < B
        ptrs = (x_ptr + b_offsets[:, None, None] * (M * N) +
                m_offsets[None, :, None] * N + n_offsets[None, None, :])
        # Use maskless fast path when tile fully inside bounds
        if (kb * BLOCK_KB + BLOCK_KB <= B) and (m_start + BLOCK_M
                                                <= M) and (n_start + BLOCK_N
                                                           <= N):
            x = tl.load(ptrs)
        else:
            x = tl.load(ptrs,
                        mask=mask_b[:, None, None] & mask_m[None, :, None]
                        & mask_n[None, None, :],
                        other=0.0)
        x = x.to(tl.float32)
        acc += tl.sum(x, axis=0)

    # store to out: shape [1, M, N] contiguous -> offset m*N + n
    out_ptrs = out_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])
