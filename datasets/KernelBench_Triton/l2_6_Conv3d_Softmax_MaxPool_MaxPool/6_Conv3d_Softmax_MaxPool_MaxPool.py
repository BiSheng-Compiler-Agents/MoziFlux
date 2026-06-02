import triton
import triton.language as tl

@triton.jit
def _softmax_pool2_fused_kernel(
    x_ptr, y_ptr,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    N, C, D, H, W, OD, OH, OW,
    K: tl.constexpr,                 # fused kernel size (K1 * K1)
    BLOCK_C: tl.constexpr,           # channel tile (>= C, padded to pow2)
    BLOCK_OW: tl.constexpr,          # width tile
):
    # Grid:
    #  axis 0 => over (N * OD * OH)
    #  axis 1 => tiles along OW
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    oh = pid0 % OH
    t = pid0 // OH
    od = t % OD
    n = t // OD

    ow_start = pid1 * BLOCK_OW
    offs_ow = tl.arange(0, BLOCK_OW)
    ow = ow_start + offs_ow
    mask_ow = ow < OW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Accumulator for the pooled result over the fused KxKxK window
    acc = tl.full([BLOCK_C, BLOCK_OW], -float("inf"), dtype=tl.float32)

    # Starting coordinates in the input for this output tile
    in_d0 = od * K
    in_h0 = oh * K
    in_w0_tile = ow_start * K

    # Base pointer for the (n, in_d0, in_h0) plane
    plane_base = n * x_stride_n + in_d0 * x_stride_d + in_h0 * x_stride_h

    # Iterate over fused K x K x K window
    for kd in range(0, K):
        for kh in range(0, K):
            # Base for this kd, kh slice
            slice_base = plane_base + kd * x_stride_d + kh * x_stride_h + in_w0_tile * x_stride_w
            # For width, each output ow pulls from input w = ow*K + kw
            ow_offsets = offs_ow * K  # [BLOCK_OW]
            for kw in range(0, K):
                # Build 2D pointers [BLOCK_C, BLOCK_OW]
                ptrs = (x_ptr
                        + slice_base
                        + ow_offsets[None, :] * x_stride_w
                        + kw * x_stride_w
                        + offs_c[:, None] * x_stride_c)
                m2d = mask_c[:, None] & mask_ow[None, :]
                x = tl.load(ptrs, mask=m2d, other=-float("inf")).to(tl.float32)
                # Channel-wise softmax for each spatial position in the tile
                x_max = tl.max(x, axis=0)                          # [BLOCK_OW]
                x = tl.exp(x - x_max[None, :])                     # [BLOCK_C, BLOCK_OW]
                x_sum = tl.sum(x, axis=0)                          # [BLOCK_OW]
                y_tile = x / x_sum[None, :]                        # [BLOCK_C, BLOCK_OW]
                # Max-pool over the fused window
                acc = tl.maximum(acc, y_tile)

    # Store results
    out_ptrs = (y_ptr
                + n * y_stride_n
                + od * y_stride_d
                + oh * y_stride_h
                + ow[None, :] * y_stride_w
                + offs_c[:, None] * y_stride_c)
    tl.store(out_ptrs, acc, mask=(mask_c[:, None] & mask_ow[None, :]))
