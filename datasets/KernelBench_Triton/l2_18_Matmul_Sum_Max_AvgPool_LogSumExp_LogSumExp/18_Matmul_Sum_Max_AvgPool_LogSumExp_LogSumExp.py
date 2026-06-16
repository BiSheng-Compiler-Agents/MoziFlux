import triton
import triton.language as tl


@triton.jit
def _rowwise_linear_sum_kernel(
    x_ptr,  # (B, I)
    wsum_ptr,  # (I,)
    out_ptr,  # (B,) result
    B: tl.constexpr,
    I: tl.constexpr,
    stride_x_b,
    stride_x_i,
    stride_wsum,
    stride_out_b,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_rows = rows < B

    # Accumulator for each row in the block
    acc = tl.zeros([BLOCK_B], dtype=tl.float32)

    # Precompute row base pointers for coalesced access
    row_ptrs = x_ptr + rows[:, None] * stride_x_b

    # Use a runtime-controlled loop (no tl.static_range) to avoid constexpr issues.
    # Unroll by 2 to reduce loop overhead while keeping correct masking.
    k_start = 0
    while k_start < I:
        # Iteration 0
        k_idx0 = k_start + tl.arange(0, BLOCK_K)
        mask_k0 = k_idx0 < I
        x_tile0 = tl.load(
            row_ptrs + k_idx0[None, :] * stride_x_i,
            mask=mask_rows[:, None] & mask_k0[None, :],
            other=0.0,
        ).to(tl.float32)
        w_tile0 = tl.load(
            wsum_ptr + k_idx0 * stride_wsum,
            mask=mask_k0,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x_tile0 * w_tile0[None, :], axis=1)

        # Iteration 1 (may be fully masked if beyond I)
        k_idx1 = k_start + BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k1 = k_idx1 < I
        x_tile1 = tl.load(
            row_ptrs + k_idx1[None, :] * stride_x_i,
            mask=mask_rows[:, None] & mask_k1[None, :],
            other=0.0,
        ).to(tl.float32)
        w_tile1 = tl.load(
            wsum_ptr + k_idx1 * stride_wsum,
            mask=mask_k1,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x_tile1 * w_tile1[None, :], axis=1)

        k_start += 2 * BLOCK_K

    # Write result
    tl.store(out_ptr + rows * stride_out_b, acc, mask=mask_rows)


@triton.jit
def _fused_linear_sum_kernel(
    x_ptr,  # *f32 (B, I)
    W_ptr,  # *f32 (O, I)
    b_ptr,  # *f32 (O,) - can be dummy if O_b==0
    out_ptr,  # *f32 (B,)
    B,
    I,
    O,  # int32 sizes
    stride_x_b,  # int32
    stride_x_i,  # int32
    stride_w_o,  # int32
    stride_w_i,  # int32
    stride_b_o,  # int32
    stride_out_b,  # int32
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    UNROLL_O: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_rows = rows < B

    # Base ptrs
    row_ptrs = x_ptr + rows[:, None] * stride_x_b

    # Accumulator per row
    acc = tl.zeros([BLOCK_B], dtype=tl.float32)

    # Precompute bias sum once per program
    c_acc = tl.zeros((), dtype=tl.float32)
    offs_o = tl.arange(0, UNROLL_O)
    o = 0
    while o < O:
        o_idx = o + offs_o
        mask_o = o_idx < O
        b_vals = tl.load(b_ptr + o_idx * stride_b_o, mask=mask_o,
                         other=0.0).to(tl.float32)
        c_acc += tl.sum(b_vals, axis=0)
        o += UNROLL_O

    # Iterate over I in tiles
    offs_k = tl.arange(0, BLOCK_K)
    k = 0
    while k < I:
        k_idx = k + offs_k
        mask_k = k_idx < I

        # Compute weight column-sum tile v[k] = sum_o W[o, k]
        v_tile = tl.zeros([BLOCK_K], dtype=tl.float32)
        o2 = 0
        while o2 < O:
            o2_idx = o2 + offs_o
            mask_o2 = o2_idx < O
            w_block = tl.load(
                W_ptr + o2_idx[:, None] * stride_w_o +
                k_idx[None, :] * stride_w_i,
                mask=mask_o2[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            v_tile += tl.sum(w_block, axis=0)
            o2 += UNROLL_O

        # Load x tile and accumulate dot for all rows in this block
        x_tile = tl.load(
            row_ptrs + k_idx[None, :] * stride_x_i,
            mask=mask_rows[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x_tile * v_tile[None, :], axis=1)

        k += BLOCK_K

    # Add bias sum and store
    acc += c_acc
    tl.store(out_ptr + rows * stride_out_b, acc, mask=mask_rows)
