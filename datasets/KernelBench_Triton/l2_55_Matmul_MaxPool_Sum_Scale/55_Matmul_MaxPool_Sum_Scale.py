import triton
import triton.language as tl


@triton.jit
def _linear_maxpool_sum_scale_kernel(
        x_ptr,  # float32[B, IN_F]
        w_ptr,  # float32[OUT_F, IN_F]
        b_ptr,  # float32[OUT_F]
        out_ptr,  # float32[B]
        B: tl.constexpr,  # batch size
        IN_F,  # in_features (int)
        OUT_F,  # out_features (int)
        KERNEL,  # kernel_size (int), stride == kernel_size
        X_STRIDE,  # stride for x rows (int)
        W_ROW_STRIDE,  # weight row stride (int) -> typically IN_F
        W_COL_STRIDE,  # weight col stride (int) -> typically 1
        SCALE,  # scale factor (float)
        BLOCK_IN: tl.constexpr,  # tile size along IN_F
        BLOCK_KO: tl.constexpr,  # tile size along kernel window outputs
):
    pid = tl.program_id(axis=0)
    # Each program handles one row (batch element)
    x_row_ptr = x_ptr + pid * X_STRIDE

    # Number of non-overlapping pooling windows
    windows_count = OUT_F // KERNEL

    sum_acc = 0.0
    win = 0
    # Iterate over windows
    while win < windows_count:
        base_o = win * KERNEL
        # Track maximum over the current window
        window_max = -float("inf")

        ko_off = 0
        # Process the K outputs of the window in chunks of BLOCK_KO
        while ko_off < KERNEL:
            offs_k = tl.arange(0, BLOCK_KO)
            o_idx = base_o + ko_off + offs_k
            mask_o = o_idx < (base_o + KERNEL)

            # Accumulators for dot products for this chunk [BLOCK_KO]
            acc = tl.zeros([BLOCK_KO], dtype=tl.float32)

            m_off = 0
            # Reduce over input features dimension
            while m_off < IN_F:
                offs_m = tl.arange(0, BLOCK_IN)
                m_idx = m_off + offs_m
                mask_m = m_idx < IN_F

                # Load x chunk [BLOCK_IN]
                x_vec = tl.load(x_row_ptr + m_idx, mask=mask_m, other=0.0)

                # Load weight sub-matrix [BLOCK_KO, BLOCK_IN]
                w_ptrs = w_ptr + (o_idx[:, None] * W_ROW_STRIDE) + (
                    m_idx[None, :] * W_COL_STRIDE)
                w_block = tl.load(w_ptrs,
                                  mask=mask_o[:, None] & mask_m[None, :],
                                  other=0.0)

                # FMA reduction over input features for each output in the window chunk
                acc += tl.sum(w_block * x_vec[None, :], axis=1)

                m_off += BLOCK_IN

            # Add bias
            b_vec = tl.load(b_ptr + o_idx, mask=mask_o, other=0.0)
            val_vec = acc + b_vec

            # Compute max over valid elements in this chunk and update window max
            chunk_max = tl.max(tl.where(mask_o, val_vec, -float("inf")),
                               axis=0)
            window_max = tl.maximum(window_max, chunk_max)

            ko_off += BLOCK_KO

        # Accumulate sum over window maxima
        sum_acc += window_max
        win += 1

    # Scale and store result
    out_val = sum_acc * SCALE
    tl.store(out_ptr + pid, out_val)
