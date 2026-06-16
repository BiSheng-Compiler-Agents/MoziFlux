import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    CIN,
    COUT,
    LIN,
    LOUT,
    K,
    STRIDE,
    PADDING,
    DILATION,
    stride_xn,
    stride_xc,
    stride_xl,
    stride_wci,
    stride_wco,
    stride_wk,
    stride_yn,
    stride_yc,
    stride_yl,
    HAS_BIAS: tl.constexpr,
    CIN_C: tl.constexpr,
    K_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    # Map program id to (n, co)
    co = pid0 % COUT
    n = pid0 // COUT

    # Offsets in output length dimension
    offs_t = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < LOUT

    # Base pointers
    y_base = y_ptr + n * stride_yn + co * stride_yc
    base_xn = x_ptr + n * stride_xn
    base_wco = w_ptr + co * stride_wco

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_T, ), dtype=tl.float32)

    # Optional bias
    if HAS_BIAS:
        b_val = tl.load(b_ptr + co)
        acc += b_val.to(tl.float32)

    # Precompute output->input mapping terms once
    i_base = offs_t + PADDING
    q = i_base // STRIDE
    r = i_base - q * STRIDE  # r = i_base % STRIDE

    # Loop over kernel taps and input channels (fully unrolled)
    for k in tl.static_range(0, K_C):
        kd = k * DILATION
        a = kd % STRIDE
        b = kd // STRIDE

        # Valid alignment when r == a
        mask_align = (r == a) & mask_t
        # Corresponding input index
        i_vec = q - b
        mask_i = mask_align & (i_vec >= 0) & (i_vec < LIN)
        # Safe offsets for loads
        i_safe = tl.where(mask_i, i_vec, 0)
        x_offs = i_safe * stride_xl

        wk_base = base_wco + k * stride_wk

        for ci in tl.static_range(0, CIN_C):
            # Load weight scalar w[ci, co, k]
            w_val = tl.load(wk_base + ci * stride_wci).to(tl.float32)

            # Load x[n, ci, i] for vector i (gather)
            x_base_ci = base_xn + ci * stride_xc
            x_vals = tl.load(x_base_ci + x_offs, mask=mask_i,
                             other=0.0).to(tl.float32)

            acc += x_vals * w_val

    # Store results
    tl.store(y_base + offs_t * stride_yl, acc, mask=mask_t)
