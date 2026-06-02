import triton
import triton.language as tl

@triton.jit
def _conv1d_fwd_kernel(
    x_ptr,           # float*        [N, IC, L_IN]
    w_ptr,           # float*        [OC, IC, K]
    b_ptr,           # float* or dummy (unused when HAS_BIAS=False) [OC]
    y_ptr,           # float*        [N, OC, L_OUT]
    N,               # int
    L_IN,            # int
    OC,              # int
    L_OUT,           # int
    x_stride_n,      # int
    x_stride_c,      # int
    x_stride_l,      # int
    w_stride_o,      # int
    w_stride_c,      # int
    w_stride_k,      # int
    y_stride_n,      # int
    y_stride_o,      # int
    y_stride_l,      # int
    STRIDE: tl.constexpr,     # int (constexpr)
    DILATION: tl.constexpr,   # int (constexpr)
    IC: tl.constexpr,         # int (constexpr)
    K: tl.constexpr,          # int (constexpr)
    HAS_BIAS: tl.constexpr,   # bool (constexpr)
    BLOCK_OC: tl.constexpr,   # tile size for OC
):
    # Program IDs over (N * L_OUT) and OC tiles (unchanged PID logic)
    pid_nl = tl.program_id(0)
    pid_ob = tl.program_id(1)

    n = pid_nl // L_OUT
    p = pid_nl % L_OUT

    # Tile of output channels handled by this program
    oc_offsets = pid_ob * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < OC

    # Accumulator for [BLOCK_OC] results in FP32
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # Base positions/pointers
    pos0 = p * STRIDE
    x_n_base = n * x_stride_n

    # Precompute base pointer per output-channel in the weights
    w_co_base = w_ptr + oc_offsets * w_stride_o

    # Reduction over input channels and kernel taps (no masked RBLOCK to reduce overhead)
    for ic in tl.static_range(0, IC):
        x_nc_base = x_ptr + x_n_base + ic * x_stride_c
        w_c_base = w_co_base + ic * w_stride_c
        for k in tl.static_range(0, K):
            t = pos0 + k * DILATION
            in_bounds = t < L_IN  # keep boundary check as required
            # Load single input value
            x_val = tl.load(x_nc_base + t * x_stride_l, mask=in_bounds, other=0.0)
            # Load vector of weights for this (ic, k) across BLOCK_OC output channels
            w_vec = tl.load(w_c_base + k * w_stride_k, mask=oc_mask, other=0.0)
            # FMA accumulate
            acc += w_vec * x_val

    # Add bias if present
    if HAS_BIAS:
        b_vec = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0)
        acc += b_vec

    # Store results
    y_idx = n * y_stride_n + oc_offsets * y_stride_o + p * y_stride_l
    tl.store(y_ptr + y_idx, acc, mask=oc_mask)
