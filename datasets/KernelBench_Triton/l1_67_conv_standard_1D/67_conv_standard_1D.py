import triton
import triton.language as tl

@triton.jit
def conv1d_fwd_kernel(
    x_ptr,  # float32[B, C, L_IN]
    w_ptr,  # float32[OC, C, K]
    y_ptr,  # float32[B, OC, L_OUT]
    B, C, L_IN, OC, K,
    STRIDE, PADDING, DILATION, L_OUT, CK,
    BLOCK_OC: tl.constexpr,  # tile size in output-channel dimension
    BLOCK_T: tl.constexpr,   # tile size in time/output-length dimension
    BLOCK_P: tl.constexpr,   # tile size in reduction dimension (C*K)
    NUM_P_ITERS: tl.constexpr,
):
    # Program IDs (PID logic must not be changed)
    pid_t = tl.program_id(0)   # tile id for output length (time)
    pid_oc = tl.program_id(1)  # tile id for output channels
    pid_b = tl.program_id(2)   # batch id

    # Tile start indices
    t_start = pid_t * BLOCK_T
    oc_start = pid_oc * BLOCK_OC
    b = pid_b

    # Indices in this tile
    t_idx = t_start + tl.arange(0, BLOCK_T)            # [BLOCK_T]
    oc_idx = oc_start + tl.arange(0, BLOCK_OC)         # [BLOCK_OC]
    tl.max_contiguous(t_idx, BLOCK_T)
    tl.max_contiguous(oc_idx, BLOCK_OC)

    # Masks that don't depend on the reduction loop
    mask_t = t_idx < L_OUT
    mask_oc = oc_idx < OC

    # Accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_T), dtype=tl.float32)

    # Helpful precomputations
    b_base = b * C * L_IN
    oc_base = oc_idx[:, None] * CK
    t_term = t_idx[None, :] * STRIDE - PADDING  # reused across iterations

    # Double-buffered software pipelining over the reduction dimension
    if NUM_P_ITERS > 0:
        ar_p = tl.arange(0, BLOCK_P)

        # Prefetch first chunk
        p0 = 0
        p_idx = p0 + ar_p                             # [BLOCK_P]
        ic_idx = p_idx // K                           # [BLOCK_P]
        k_idx = p_idx % K                             # [BLOCK_P]

        pos = t_term + k_idx[:, None] * DILATION  # [BLOCK_P, BLOCK_T]
        mask_x = (p_idx < CK)[:, None] & (pos >= 0) & (pos < L_IN)

        x_offsets = b_base + ic_idx[:, None] * L_IN + pos
        w_offsets = oc_base + ic_idx[None, :] * K + k_idx[None, :]

        mask_p = p_idx < CK
        mask_w = mask_oc[:, None] & mask_p[None, :]

        x_tile = tl.load(x_ptr + x_offsets, mask=mask_x, other=0.0)  # [BLOCK_P, BLOCK_T]
        w_tile = tl.load(w_ptr + w_offsets, mask=mask_w, other=0.0)  # [BLOCK_OC, BLOCK_P]

        # Iterate remaining chunks with prefetch of next
        for it in tl.static_range(1, NUM_P_ITERS):
            p0 = it * BLOCK_P
            p_idx_n = p0 + ar_p
            ic_idx_n = p_idx_n // K
            k_idx_n = p_idx_n % K

            pos_n = t_term + k_idx_n[:, None] * DILATION
            mask_x_n = (p_idx_n < CK)[:, None] & (pos_n >= 0) & (pos_n < L_IN)

            x_offsets_n = b_base + ic_idx_n[:, None] * L_IN + pos_n
            w_offsets_n = oc_base + ic_idx_n[None, :] * K + k_idx_n[None, :]

            mask_p_n = p_idx_n < CK
            mask_w_n = mask_oc[:, None] & mask_p_n[None, :]

            # Prefetch next tiles
            x_next = tl.load(x_ptr + x_offsets_n, mask=mask_x_n, other=0.0)
            w_next = tl.load(w_ptr + w_offsets_n, mask=mask_w_n, other=0.0)

            # Compute on the current tiles while next ones are being fetched
            acc += tl.dot(w_tile, x_tile)

            # Swap buffers
            x_tile = x_next
            w_tile = w_next

        # Final accumulated dot
        acc += tl.dot(w_tile, x_tile)

    # Store results
    y_offsets = b * OC * L_OUT + oc_idx[:, None] * L_OUT + t_idx[None, :]
    mask_y = mask_oc[:, None] & mask_t[None, :]
    tl.store(y_ptr + y_offsets, acc, mask=mask_y)
