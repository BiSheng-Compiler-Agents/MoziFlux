import triton
import triton.language as tl

@triton.jit
def _dwconv2d_kernel(
    x_ptr,        # *f32 [N, C, H, W]
    w_ptr,        # *f32 [C, 1, K, K]
    b_ptr,        # *f32 [C] or dummy
    y_ptr,        # *f32 [N, C, H_OUT, W_OUT]
    N, C, H, W,
    H_OUT, W_OUT,
    S: tl.constexpr,     # stride (square)
    P: tl.constexpr,     # padding (square)
    K: tl.constexpr,     # kernel size (square)
    stride_xN, stride_xC, stride_xH, stride_xW,
    stride_wC, stride_wH, stride_wW,
    stride_yN, stride_yC, stride_yH, stride_yW,
    HAS_BIAS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_nc = tl.program_id(0)      # over N*C
    pid_h = tl.program_id(1)       # over H_OUT
    pid_w = tl.program_id(2)       # over W_OUT tiles

    # derive n, c from pid_nc
    n = pid_nc // C
    c = pid_nc % C

    oh = pid_h  # one row per program on axis-1

    w_start = pid_w * BLOCK_W
    ow = w_start + tl.arange(0, BLOCK_W)
    out_mask = ow < W_OUT

    # base pointers/offsets
    y_base = n * stride_yN + c * stride_yC + oh * stride_yH
    x_plane_base = n * stride_xN + c * stride_xC

    # initialize accumulator
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # compute input top-left coordinate for this output row
    ih0 = oh * S - P
    iw0 = ow * S - P
    w_base_c = c * stride_wC

    # Fast path for common case: stride=1, padding=0 -> no interior bound checks needed
    if (S == 1) & (P == 0):
        # For this case, for all lanes with ow < W_OUT, all KxK taps are guaranteed in-bounds.
        for kh in tl.static_range(0, K):
            ih = oh + kh  # valid by construction for all lanes with out_mask
            x_row_ptr = x_ptr + x_plane_base + ih * stride_xH
            w_row_base = w_base_c + kh * stride_wH

            # start pointer for kw=0
            x_ptrs = x_row_ptr + iw0 * stride_xW
            # unrolled over kernel width with pointer bumping
            for kw in tl.static_range(0, K):
                x_val = tl.load(x_ptrs, mask=out_mask, other=0.0)
                w_val = tl.load(w_ptr + w_row_base + kw * stride_wW)
                acc += x_val * w_val
                x_ptrs += stride_xW
    else:
        # General path with full boundary checks
        for kh in tl.static_range(0, K):
            ih = ih0 + kh
            valid_h = (ih >= 0) & (ih < H)

            x_row_ptr = x_ptr + x_plane_base + ih * stride_xH
            w_row_base = w_base_c + kh * stride_wH

            # start pointer for kw=0 and advance each step
            x_ptrs = x_row_ptr + iw0 * stride_xW
            for kw in tl.static_range(0, K):
                # width validity for this kw
                valid_w = (iw0 + kw >= 0) & (iw0 + kw < W)
                mask = out_mask & valid_h & valid_w

                x_val = tl.load(x_ptrs, mask=mask, other=0.0)
                w_val = tl.load(w_ptr + w_row_base + kw * stride_wW)
                acc += x_val * w_val

                x_ptrs += stride_xW

    # add bias if present
    if HAS_BIAS:
        b_val = tl.load(b_ptr + c)
        acc += b_val

    # store result
    y_ptrs = y_ptr + y_base + ow * stride_yW
    tl.store(y_ptrs, acc, mask=out_mask)
