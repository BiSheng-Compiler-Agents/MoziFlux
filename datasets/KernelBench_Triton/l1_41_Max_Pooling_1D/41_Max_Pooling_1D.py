import triton
import triton.language as tl

@triton.jit
def _maxpool1d_forward_kernel(
    x_ptr,                # *T,  input [NC][L_in]
    y_ptr,                # *T,  output [NC][L_out]
    idx_ptr,              # *int64, indices [NC][L_out] (optional)
    L_in,                 # int32
    L_out,                # int32
    STRIDE,               # int32
    PADDING,              # int32
    DILATION,             # int32
    line_stride_x,        # int32 = L_in
    line_stride_y,        # int32 = L_out
    HAS_INDEX: tl.constexpr,  # bool, whether to write indices
    K: tl.constexpr,          # kernel size (compile-time)
    BLOCK: tl.constexpr,      # tile size along output length
):
    pid_nc = tl.program_id(axis=0)          # which (N,C) line
    pid_o_blk = tl.program_id(axis=1)       # which output tile

    o_offsets = pid_o_blk * BLOCK + tl.arange(0, BLOCK)
    mask_o = o_offsets < L_out

    # compute window start per output position
    starts = o_offsets * STRIDE - PADDING  # [BLOCK], int32

    base_x = x_ptr + pid_nc * line_stride_x
    base_y = y_ptr + pid_nc * line_stride_y

    # Vectorized, mostly-unmasked loads with clamped addresses to improve coalescing.
    pos = starts
    valid0 = (pos >= 0) & (pos < L_in) & mask_o
    # clamp to valid range to allow unmasked loads for valid output lanes
    addr0 = tl.minimum(tl.maximum(pos, 0), L_in - 1)
    x0 = tl.load(base_x + addr0, mask=mask_o, other=0)
    y_max = tl.where(valid0, x0, -float("inf"))
    if HAS_INDEX:
        chosen_pos = pos

    # k = 1..K-1
    for _ in tl.static_range(1, K):
        pos = pos + DILATION
        validk = (pos >= 0) & (pos < L_in) & mask_o
        addrk = tl.minimum(tl.maximum(pos, 0), L_in - 1)
        xk = tl.load(base_x + addrk, mask=mask_o, other=0)
        vk = tl.where(validk, xk, -float("inf"))
        better = vk > y_max
        y_max = tl.where(better, vk, y_max)
        if HAS_INDEX:
            chosen_pos = tl.where(better, pos, chosen_pos)

    # Store results
    tl.store(base_y + o_offsets, y_max, mask=mask_o)

    if HAS_INDEX:
        # Clamp to valid input range; out-of-bounds lanes end up clamped to 0.
        chosen_pos = tl.maximum(0, tl.minimum(chosen_pos, L_in - 1))
        base_i = idx_ptr + pid_nc * line_stride_y
        tl.store(base_i + o_offsets, chosen_pos.to(tl.int64), mask=mask_o)
