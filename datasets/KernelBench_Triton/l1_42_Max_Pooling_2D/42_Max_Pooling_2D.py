import triton
import triton.language as tl


@triton.jit
def _maxpool2d_kernel(
    x_ptr,  # *fp*  [N, C, H, W] contiguous
    y_ptr,  # *fp*  [N, C, H_out, W_out] contiguous
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    GRID_WO: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    BLOCK_WO: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_ho = pid_hw // GRID_WO
    pid_wo = pid_hw % GRID_WO

    # Derive n and c from flattened nc
    n = pid_nc // C
    c = pid_nc % C

    # Tiled output indices
    ho_offsets = pid_ho * BLOCK_HO + tl.arange(0, BLOCK_HO)
    wo_offsets = pid_wo * BLOCK_WO + tl.arange(0, BLOCK_WO)

    ho_mask = ho_offsets < H_out
    wo_mask = wo_offsets < W_out

    HO = ho_offsets[:, None]  # [BH, 1]
    WO = wo_offsets[None, :]  # [1, BW]
    out_mask = ho_mask[:, None] & wo_mask[None, :]  # [BH, BW]

    # Strides for contiguous NCHW layout
    HW = H * W
    HWo = H_out * W_out
    base_x = (n * C + c) * HW
    base_y = (n * C + c) * HWo

    # Precompute output offsets
    y_offs = base_y + HO * W_out + WO

    # Input top-left start for each (ho, wo)
    h_start = HO * STRIDE_H - PAD_H  # [BH, BW]
    w_start = WO * STRIDE_W - PAD_W  # [BH, BW]

    # Detect "interior tiles" where all pooling windows are in-bounds and the tile is fully inside output
    # This allows us to skip masks entirely for the common case.
    tile_full_ho = ((pid_ho + 1) * BLOCK_HO) <= H_out
    tile_full_wo = ((pid_wo + 1) * BLOCK_WO) <= W_out
    hs0 = pid_ho * BLOCK_HO * STRIDE_H - PAD_H
    ws0 = pid_wo * BLOCK_WO * STRIDE_W - PAD_W
    hs_last = hs0 + (BLOCK_HO - 1) * STRIDE_H + (K_H - 1) * DIL_H
    ws_last = ws0 + (BLOCK_WO - 1) * STRIDE_W + (K_W - 1) * DIL_W
    tile_interior = tile_full_ho & tile_full_wo & (hs0 >= 0) & (
        hs_last < H) & (ws0 >= 0) & (ws_last < W)

    exact_4x4_stride1 = ((K_H == 4) and (K_W == 4) and (STRIDE_H == 1)
                         and (STRIDE_W == 1) and (PAD_H == 1) and (PAD_W == 1)
                         and (DIL_H == 1) and (DIL_W == 1))

    if exact_4x4_stride1:
        if tile_interior:
            row0 = base_x + h_start * W
            row1 = row0 + W
            row2 = row1 + W
            row3 = row2 + W

            v00 = tl.load(x_ptr + row0 + w_start)
            v01 = tl.load(x_ptr + row0 + w_start + 1)
            v02 = tl.load(x_ptr + row0 + w_start + 2)
            v03 = tl.load(x_ptr + row0 + w_start + 3)
            v10 = tl.load(x_ptr + row1 + w_start)
            v11 = tl.load(x_ptr + row1 + w_start + 1)
            v12 = tl.load(x_ptr + row1 + w_start + 2)
            v13 = tl.load(x_ptr + row1 + w_start + 3)
            v20 = tl.load(x_ptr + row2 + w_start)
            v21 = tl.load(x_ptr + row2 + w_start + 1)
            v22 = tl.load(x_ptr + row2 + w_start + 2)
            v23 = tl.load(x_ptr + row2 + w_start + 3)
            v30 = tl.load(x_ptr + row3 + w_start)
            v31 = tl.load(x_ptr + row3 + w_start + 1)
            v32 = tl.load(x_ptr + row3 + w_start + 2)
            v33 = tl.load(x_ptr + row3 + w_start + 3)

            r0 = tl.maximum(tl.maximum(v00, v01), tl.maximum(v02, v03))
            r1 = tl.maximum(tl.maximum(v10, v11), tl.maximum(v12, v13))
            r2 = tl.maximum(tl.maximum(v20, v21), tl.maximum(v22, v23))
            r3 = tl.maximum(tl.maximum(v30, v31), tl.maximum(v32, v33))
            max_val = tl.maximum(tl.maximum(r0, r1), tl.maximum(r2, r3))
            tl.store(y_ptr + y_offs, max_val)
        else:
            max_val = tl.full((BLOCK_HO, BLOCK_WO), -float("inf"), tl.float32)
            for kh in tl.static_range(0, 4):
                ih = h_start + kh
                ih_in = (ih >= 0) & (ih < H)
                safe_ih = tl.where(ih_in, ih, 0)
                row_base = base_x + safe_ih * W
                for kw in tl.static_range(0, 4):
                    iw = w_start + kw
                    iw_in = (iw >= 0) & (iw < W)
                    in_bounds = out_mask & ih_in & iw_in
                    safe_iw = tl.where(iw_in, iw, 0)
                    val = tl.load(x_ptr + row_base + safe_iw,
                                  mask=in_bounds,
                                  other=-float("inf"))
                    max_val = tl.maximum(max_val, val)

            tl.store(y_ptr + y_offs, max_val, mask=out_mask)
    # Fastpath for very common 2x2 kernels
    elif (K_H == 2) and (K_W == 2):
        ih0 = h_start
        ih1 = ih0 + DIL_H
        iw0 = w_start
        iw1 = iw0 + DIL_W

        step_h = DIL_H * W
        step_w = DIL_W

        if tile_interior:
            # Fully interior: no masks needed
            row0_base = base_x + ih0 * W
            offs00 = row0_base + iw0
            v00 = tl.load(x_ptr + offs00)
            v01 = tl.load(x_ptr + (offs00 + step_w))

            row1_base = row0_base + step_h
            offs10 = row1_base + iw0
            v10 = tl.load(x_ptr + offs10)
            v11 = tl.load(x_ptr + (offs10 + step_w))

            m0 = tl.maximum(v00, v01)
            m1 = tl.maximum(v10, v11)
            max_val = tl.maximum(m0, m1)
            tl.store(y_ptr + y_offs, max_val)
        else:
            ih0_in = (ih0 >= 0) & (ih0 < H)
            ih1_in = (ih1 >= 0) & (ih1 < H)
            iw0_in = (iw0 >= 0) & (iw0 < W)
            iw1_in = (iw1 >= 0) & (iw1 < W)

            # Row-wise masks to reduce logical ops
            mask_r0 = out_mask & ih0_in
            mask_r1 = out_mask & ih1_in

            row0_base = base_x + ih0 * W
            offs00 = row0_base + iw0

            v00 = tl.load(x_ptr + offs00,
                          mask=(mask_r0 & iw0_in),
                          other=-float("inf"))
            v01 = tl.load(x_ptr + (offs00 + step_w),
                          mask=(mask_r0 & iw1_in),
                          other=-float("inf"))

            row1_base = row0_base + step_h
            offs10 = row1_base + iw0
            v10 = tl.load(x_ptr + offs10,
                          mask=(mask_r1 & iw0_in),
                          other=-float("inf"))
            v11 = tl.load(x_ptr + (offs10 + step_w),
                          mask=(mask_r1 & iw1_in),
                          other=-float("inf"))

            m0 = tl.maximum(v00, v01)
            m1 = tl.maximum(v10, v11)
            max_val = tl.maximum(m0, m1)
            tl.store(y_ptr + y_offs, max_val, mask=out_mask)
    else:
        max_val = tl.full((BLOCK_HO, BLOCK_WO), -float("inf"), tl.float32)
        for kh in tl.static_range(0, K_H):
            ih = h_start + kh * DIL_H
            ih_in = (ih >= 0) & (ih < H)
            safe_ih = tl.where(ih_in, ih, 0)
            row_base = base_x + safe_ih * W
            for kw in tl.static_range(0, K_W):
                iw = w_start + kw * DIL_W
                iw_in = (iw >= 0) & (iw < W)
                in_bounds = out_mask & ih_in & iw_in
                safe_iw = tl.where(iw_in, iw, 0)
                offs = row_base + safe_iw
                val = tl.load(x_ptr + offs,
                              mask=in_bounds,
                              other=-float("inf"))
                max_val = tl.maximum(max_val, val)

        tl.store(y_ptr + y_offs, max_val, mask=out_mask)
