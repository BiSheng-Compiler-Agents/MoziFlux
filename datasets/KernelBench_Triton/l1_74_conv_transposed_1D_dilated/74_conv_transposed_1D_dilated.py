import triton
import triton.language as tl


@triton.jit
def _conv_transpose1d_kernel(
        x_ptr,  # *f32, [B, Cin, Lin]
        w_ptr,  # *f32, [Cin, Cout, K]
        bias_ptr,  # *f32, [Cout] (optional)
        y_ptr,  # *f32, [B, Cout, Lout]
        B: tl.constexpr,  # batch size
        Cin: tl.constexpr,  # in channels
        Cout,  # out channels
        Lin,  # input length
        Lout,  # output length
        K: tl.constexpr,  # kernel size
        STRIDE: tl.constexpr,  # stride
        PADDING: tl.constexpr,  # padding
        DILATION: tl.constexpr,  # dilation
        HAS_BIAS: tl.constexpr,  # whether to add bias
        BLOCK_COUT: tl.constexpr,  # tile size along out-channels
        BLOCK_T: tl.constexpr,  # tile size along time dimension
):
    # program ids
    pid0 = tl.program_id(axis=0)  # over (B, T-blocks)
    pid1 = tl.program_id(axis=1)  # over Cout blocks

    # decompose pid0 into batch and time-block id
    T_BLOCKS = tl.cdiv(Lout, BLOCK_T)
    b = pid0 // T_BLOCKS
    tb = pid0 % T_BLOCKS

    # tile offsets
    oc_offsets = pid1 * BLOCK_COUT + tl.arange(0, BLOCK_COUT)
    t_offsets = tb * BLOCK_T + tl.arange(0, BLOCK_T)

    oc_mask = oc_offsets < Cout
    t_mask = t_offsets < Lout

    # accumulators
    acc = tl.zeros((BLOCK_COUT, BLOCK_T), dtype=tl.float32)

    # base pointers per batch
    x_batch_base = (b * Cin) * Lin
    y_batch_base = (b * Cout) * Lout

    # Precompute base for stride==1 path
    base_t = t_offsets + PADDING

    # Loop over in-channels and kernel elements (compile-time unrolled)
    for ic in tl.static_range(0, Cin):
        x_ic_base = x_batch_base + ic * Lin
        w_ic_base = (ic * Cout) * K
        w_ptr_ic = w_ptr + w_ic_base + oc_offsets * K

        if STRIDE == 1:
            # Fast path: no div/mod
            for k in tl.static_range(0, K):
                t_in = base_t - k * DILATION
                vmask = (t_in >= 0) & (t_in < Lin) & t_mask
                t_in_safe = tl.where(vmask, t_in, 0)
                x_vals = tl.load(x_ptr + x_ic_base + t_in_safe,
                                 mask=vmask,
                                 other=0.0).to(tl.float32)
                w_vec = tl.load(w_ptr_ic + k, mask=oc_mask,
                                other=0.0).to(tl.float32)
                acc += w_vec[:, None] * x_vals[None, :]
        else:
            # General path: require divisibility by stride
            for k in tl.static_range(0, K):
                n_vec = base_t - k * DILATION
                div_ok = (n_vec % STRIDE) == 0
                t_in = n_vec // STRIDE
                vmask = div_ok & (t_in >= 0) & (t_in < Lin) & t_mask
                t_in_safe = tl.where(vmask, t_in, 0)
                x_vals = tl.load(x_ptr + x_ic_base + t_in_safe,
                                 mask=vmask,
                                 other=0.0).to(tl.float32)
                w_vec = tl.load(w_ptr_ic + k, mask=oc_mask,
                                other=0.0).to(tl.float32)
                acc += w_vec[:, None] * x_vals[None, :]

    # Add bias if present
    if HAS_BIAS:
        b_vec = tl.load(bias_ptr + oc_offsets, mask=oc_mask,
                        other=0.0).to(tl.float32)
        acc += b_vec[:, None]

    # Store results
    y_idx = y_batch_base + oc_offsets[:, None] * Lout + t_offsets[None, :]
    mask_2d = oc_mask[:, None] & t_mask[None, :]
    tl.store(y_ptr + y_idx, acc, mask=mask_2d)
