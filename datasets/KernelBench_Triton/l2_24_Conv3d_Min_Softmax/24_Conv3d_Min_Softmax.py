import triton
import triton.language as tl

@triton.jit
def _min_reduce_dim2_kernel(
    x_ptr,                # *f32 [B, C, D, H, W]
    y_ptr,                # *f32 [B, C, H, W]
    B, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_W: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid maps over (n, c, h, w_tile)
    pid = tl.program_id(axis=0)
    num_w_tiles = tl.cdiv(W, BLOCK_W)

    w_tile = pid % num_w_tiles
    pid = pid // num_w_tiles
    h_idx = pid % H
    pid = pid // H
    c_idx = pid % C
    n_idx = pid // C

    w_offsets = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
    d_offsets = tl.arange(0, BLOCK_D)

    offs_w = w_offsets[None, :]
    offs_d = d_offsets[:, None]

    in_base = n_idx * stride_n + c_idx * stride_c + h_idx * stride_h
    ptrs = x_ptr + in_base + offs_d * stride_d + offs_w * stride_w

    mask = (w_offsets[None, :] < W) & (d_offsets[:, None] < D)
    vals = tl.load(ptrs, mask=mask, other=float("inf"))
    # Compute min across D dimension (axis=0)
    min_vals = tl.min(vals, axis=0)

    out_ptrs = y_ptr + n_idx * out_stride_n + c_idx * out_stride_c + h_idx * out_stride_h + w_offsets * out_stride_w
    tl.store(out_ptrs, min_vals, mask=w_offsets < W)

@triton.jit
def _softmax_dim1_kernel(
    x_ptr,                # *f32 [B, C, H, W]
    y_ptr,                # *f32 [B, C, H, W]
    B, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_C: tl.constexpr,
):
    # Grid maps over (n, h, w)
    pid = tl.program_id(axis=0)
    w_idx = pid % W
    pid = pid // W
    h_idx = pid % H
    n_idx = pid // H

    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    base = n_idx * stride_n + h_idx * stride_h + w_idx * stride_w
    x_ptrs = x_ptr + base + c_offsets * stride_c

    x_vals = tl.load(x_ptrs, mask=c_mask, other=-float("inf"))
    x_vals = x_vals.to(tl.float32)
    x_max = tl.max(x_vals, axis=0)
    x_vals = x_vals - x_max
    x_exp = tl.exp(x_vals)
    x_sum = tl.sum(x_exp, axis=0)
    y_vals = x_exp / x_sum

    y_ptrs = y_ptr + base + c_offsets * out_stride_c
    tl.store(y_ptrs, y_vals, mask=c_mask)

@triton.jit
def _fused_minD_softmaxC_wtile_singleCTile(
    x_ptr,                # *f32 [B, C, D, H, W]
    y_ptr,                # *f32 [B, C, H, W]
    B, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    TOT_D_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid maps over (n, h, w_tile)
    pid = tl.program_id(axis=0)
    num_w_tiles = tl.cdiv(W, BLOCK_W)

    w_tile = pid % num_w_tiles
    pid = pid // num_w_tiles
    h_idx = pid % H
    n_idx = pid // H

    # Offsets
    c_offsets = tl.arange(0, BLOCK_C)
    w_offsets = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)

    c_mask = c_offsets < C
    w_mask = w_offsets < W

    # Prepare run_min over D for each (c, w) in the tile
    pos_inf = float("inf")
    neg_inf = -float("inf")
    run_min = tl.full([BLOCK_C, BLOCK_W], pos_inf, dtype=tl.float32)

    in_base = n_idx * stride_n + h_idx * stride_h

    # Iterate over D tiles
    for dt in tl.static_range(0, TOT_D_TILES):
        d_offsets = dt * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        ptrs = (
            x_ptr
            + in_base
            + c_offsets[:, None, None] * stride_c
            + d_offsets[None, :, None] * stride_d
            + w_offsets[None, None, :] * stride_w
        )
        mask3d = c_mask[:, None, None] & d_mask[None, :, None] & w_mask[None, None, :]
        vals = tl.load(ptrs, mask=mask3d, other=pos_inf)
        tile_min = tl.min(vals, axis=1)  # reduce along D
        run_min = tl.minimum(run_min, tile_min)

    # Softmax along channel dimension per (w) with numerical stability
    gmax = tl.max(tl.where(c_mask[:, None], run_min, neg_inf), axis=0)
    exps = tl.exp(run_min - gmax[None, :])
    exps = tl.where(c_mask[:, None] & w_mask[None, :], exps, 0.0)
    gsum = tl.sum(exps, axis=0)
    out_vals = exps / gsum[None, :]

    # Store results to [B, C, H, W]
    out_ptrs = (
        y_ptr
        + n_idx * out_stride_n
        + c_offsets[:, None] * out_stride_c
        + h_idx * out_stride_h
        + w_offsets[None, :] * out_stride_w
    )
    tl.store(out_ptrs, out_vals, mask=c_mask[:, None] & w_mask[None, :])
