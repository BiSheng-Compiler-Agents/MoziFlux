import triton
import triton.language as tl

@triton.jit
def _softmax_bias_scale_sigmoid_1d(
    x_ptr,        # *f32, [N, C, H, W]
    bias_ptr,     # *f32, [C, 1, 1]
    y_ptr,        # *f32, [N, C, H, W]
    stride_n, stride_c, stride_h, stride_w,  # strides for NCHW
    b_stride_c,                              # bias stride along C
    N, C, H, W,                              # sizes
    scaling,                                  # float
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (n, h, w)
    pid = tl.program_id(axis=0)
    HW = H * W
    n = pid // HW
    hw = pid - n * HW
    h = hw // W
    w = hw - h * W

    base = n * stride_n + h * stride_h + w * stride_w

    c_idx = tl.arange(0, BLOCK_SIZE)
    mask = c_idx < C
    x_offsets = base + c_idx * stride_c

    # Load inputs; use L2-prefetch (cg) as this is streaming
    x = tl.load(x_ptr + x_offsets, mask=mask, other=-float("inf"), cache_modifier=".cg").to(tl.float32)

    # Stable softmax across channel dimension
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    denom = tl.sum(x_exp, axis=0)
    inv_denom = 1.0 / denom
    sm = x_exp * inv_denom

    # Bias add, scale, sigmoid (pre-scale bias and use FMA)
    s = tl.full((), scaling, tl.float32)
    b = tl.load(bias_ptr + c_idx * b_stride_c, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
    b_scaled = b * s
    z = tl.fma(sm, s, b_scaled)
    out = 1.0 / (1.0 + tl.exp(-z))
    tl.store(y_ptr + x_offsets, out, mask=mask)

@triton.jit
def _softmax_bias_scale_sigmoid_tiled_cw(
    x_ptr,        # *f32, [N, C, H, W]
    bias_ptr,     # *f32, [C, 1, 1]
    y_ptr,        # *f32, [N, C, H, W]
    stride_n, stride_c, stride_h, stride_w,  # strides for NCHW
    b_stride_c,                              # bias stride along C
    N, C, H, W,                              # sizes
    scaling,                                  # float
    BLOCK_C: tl.constexpr,                    # tile in C (channels)
    BLOCK_W: tl.constexpr,                    # tile in W (contiguous)
):
    # 2D grid:
    #  - axis 0: over (n, h) pairs
    #  - axis 1: over tiles of W
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    n = pid0 // H
    h = pid0 - n * H
    w_start = pid1 * BLOCK_W

    # Offsets in W (contiguous in memory)
    w_idx = w_start + tl.arange(0, BLOCK_W)
    w_mask = w_idx < W

    # Base (n, h) offset
    base = n * stride_n + h * stride_h

    # Channel tile indices
    c_idx = tl.arange(0, BLOCK_C)
    c_mask = c_idx < C

    # Pointers for a [BLOCK_C, BLOCK_W] tile
    ptrs = base + c_idx[:, None] * stride_c + w_idx[None, :] * stride_w
    # Load a full [C, Wtile] slab once into registers
    x_tile = tl.load(x_ptr + ptrs, mask=c_mask[:, None] & w_mask[None, :], other=-float("inf"), cache_modifier=".cg").to(tl.float32)

    # Softmax along channel dimension for each column independently
    m = tl.max(x_tile, axis=0)                                 # [BLOCK_W]
    x_exp = tl.exp(x_tile - m[None, :])                        # [BLOCK_C, BLOCK_W]
    denom = tl.sum(x_exp, axis=0)                              # [BLOCK_W]
    inv_denom = 1.0 / denom
    sm = x_exp * inv_denom[None, :]                            # [BLOCK_C, BLOCK_W]

    # Load bias once per channel and broadcast along W tile
    s = tl.full((), scaling, tl.float32)
    b = tl.load(bias_ptr + c_idx * b_stride_c, mask=c_mask, other=0.0, cache_modifier=".ca").to(tl.float32)  # [BLOCK_C]
    b_scaled = b * s
    z = tl.fma(sm, s, b_scaled[:, None])
    out = 1.0 / (1.0 + tl.exp(-z))

    # Store results using the same pointers
    tl.store(y_ptr + ptrs, out, mask=c_mask[:, None] & w_mask[None, :])

@triton.jit
def _softmax_bias_scale_sigmoid_nchw(
    x_ptr,        # *f32, [N, C, H, W]
    bias_ptr,     # *f32, [C, 1, 1]
    y_ptr,        # *f32, [N, C, H, W]
    stride_n, stride_c, stride_h, stride_w,  # strides for NCHW
    b_stride_c,                              # bias stride along C
    N, C: tl.constexpr, H, W,                # sizes (C constexpr to enable compile-time specialization)
    scaling,                                  # float
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (n, h, w)
    HW = H * W
    n = pid // HW
    hw = pid - n * HW
    h = hw // W
    w = hw - h * W

    # Base pointer offset for this (n, h, w) row across channels
    base = n * stride_n + h * stride_h + w * stride_w

    # Channel indices for the block
    c_idx = tl.arange(0, BLOCK_SIZE)
    mask = c_idx < C

    # Offsets for input/output along C
    x_offsets = base + c_idx * stride_c
    # Load input once; masked lanes get -inf so they don't affect reductions
    x = tl.load(x_ptr + x_offsets, mask=mask, other=-float("inf"), cache_modifier=".cg").to(tl.float32)

    # Stable softmax along C
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    denom = tl.sum(x_exp, axis=0)
    inv_denom = 1.0 / denom
    sm = x_exp * inv_denom

    # Load bias per channel
    b_offsets = c_idx * b_stride_c
    b = tl.load(bias_ptr + b_offsets, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)

    # Scale and sigmoid (FMA)
    s = tl.full((), scaling, tl.float32)
    z = tl.fma(sm, s, b * s)
    out = 1.0 / (1.0 + tl.exp(-z))

    # Store result
    tl.store(y_ptr + x_offsets, out, mask=mask)
