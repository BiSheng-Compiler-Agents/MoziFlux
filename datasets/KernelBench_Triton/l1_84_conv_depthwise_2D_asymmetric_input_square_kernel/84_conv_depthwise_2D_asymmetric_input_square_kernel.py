import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Match grid's BLOCK_W=128 to ensure full coverage without changing grid logic
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
    ],
    key=['W_OUT'],
)
@triton.jit
def _dwconv2d_kernel(
        x_ptr,  # *const T, [N, C_in, H_in, W_in]
        w_ptr,  # *const T, [C_out, 1, K, K]
        b_ptr,  # *const T or nullptr if no bias, [C_out]
        y_ptr,  # *mut T,   [N, C_out, H_out, W_out]
        N: tl.constexpr,
        C_IN,
        C_OUT,
        H_IN,
        W_IN,
        H_OUT,
        W_OUT,
        STRIDE,
        PADDING,
        OCPG,  # out_channels per group (= out_channels // in_channels)
        K: tl.constexpr,  # kernel size (square)
        HAS_BIAS: tl.constexpr,  # compile-time flag
        BLOCK_W: tl.constexpr,  # tile size along W_out
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    # Decompose pid across (N, C_OUT)
    n = pid_nc // C_OUT
    oc = pid_nc % C_OUT
    ho = pid_h

    # Tile of output width this program computes
    w_start = pid_w * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    w_mask = w_offsets < W_OUT

    # Map output channel to its input channel (depthwise groups = in_channels)
    ic = oc // OCPG

    # Accumulator
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Precompute base scales for pointer arithmetic (int64 to avoid overflow)
    n = tl.full((), n, tl.int64)
    oc_i64 = tl.full((), oc, tl.int64)
    ic_i64 = tl.full((), ic, tl.int64)
    C_IN = tl.full((), C_IN, tl.int64)
    C_OUT = tl.full((), C_OUT, tl.int64)
    H_IN = tl.full((), H_IN, tl.int64)
    W_IN = tl.full((), W_IN, tl.int64)
    H_OUT = tl.full((), H_OUT, tl.int64)
    W_OUT_i64 = tl.full((), W_OUT, tl.int64)
    STRIDE = tl.full((), STRIDE, tl.int32)
    PADDING = tl.full((), PADDING, tl.int32)

    # Compute once per-row
    ho_i = ho * STRIDE

    # Base weight offset for this output channel
    w_oc_base = oc_i64 * (K * K)

    # Loop over kernel KxK
    for r in range(K):
        hi = ho_i - PADDING + r
        hi_in = (hi >= 0) & (hi < H_IN.to(tl.int32))
        # Base offset for this (n, ic, hi, :)
        base_ncih = (((n * C_IN + ic_i64) * H_IN) + hi.to(tl.int64)) * W_IN

        for s in range(K):
            wi = w_offsets * STRIDE - PADDING + s  # vector
            in_bounds_w = (wi >= 0) & (wi < W_IN.to(tl.int32))
            mask = w_mask & hi_in & in_bounds_w

            # Load input values
            wi_i64 = wi.to(tl.int64)
            x_offsets = base_ncih + wi_i64
            x_vals = tl.load(x_ptr + x_offsets, mask=mask,
                             other=0).to(tl.float32)

            # Load weight scalar for (oc, r, s)
            w_offset = w_oc_base + (r * K + s)
            w_val = tl.load(w_ptr + w_offset).to(tl.float32)

            acc += x_vals * w_val

    if HAS_BIAS:
        b_val = tl.load(b_ptr + oc_i64).to(tl.float32)
        acc += b_val

    # Store output
    y_base = (((n * C_OUT + oc_i64) * H_OUT) + ho.to(tl.int64)) * W_OUT_i64
    y_offsets = y_base + w_offsets.to(tl.int64)
    tl.store(y_ptr + y_offsets, acc, mask=w_mask)
