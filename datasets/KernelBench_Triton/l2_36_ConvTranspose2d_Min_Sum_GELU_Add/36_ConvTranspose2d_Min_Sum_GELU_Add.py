import triton
import triton.language as tl

@triton.jit
def _fused_min_sum_gelu_add_bias(
    x_ptr,            # *f32, [N, C, H, W]
    bias_ptr,         # *f32, [C, 1, 1]
    y_ptr,            # *f32, [N, C, 1, W]
    N: tl.constexpr,  # int
    C: tl.constexpr,  # int
    H: tl.constexpr,  # int
    W: tl.constexpr,  # int
    sxn, sxc, sxh, sxw,    # strides for x
    sbc,                   # stride for bias along C
    syn, syc, syh, syw,    # strides for y
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w_blk = tl.program_id(1)
    pid_cblk = tl.program_id(2)

    # Offsets and masks along width for this program
    w_offsets = pid_w_blk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Channel tile handled by this program id
    c_offsets = pid_cblk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C

    # Base offsets for input/output batch n
    x_base_n = pid_n * sxn
    out_base_n = pid_n * syn

    # Precompute w pointer increments to save MADs in inner loops
    w_ptrs = w_offsets * sxw
    tl.multiple_of(w_ptrs, values=1)

    # Accumulator over H of per-(min over C)
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    INF = 1.0e20

    # Unroll height by 4 for better ILP
    h = 0
    while (h + 3) < H:
        # Initialize per-height minima
        min0 = tl.full((BLOCK_W,), INF, dtype=tl.float32)
        min1 = tl.full((BLOCK_W,), INF, dtype=tl.float32)
        min2 = tl.full((BLOCK_W,), INF, dtype=tl.float32)
        min3 = tl.full((BLOCK_W,), INF, dtype=tl.float32)

        c_start = 0
        while c_start < C:
            c_tile = c_start + tl.arange(0, BLOCK_C)
            mask_ct = c_tile < C
            # Base [C_tile, W_tile] pointer (independent of h)
            base_cw = (
                x_ptr
                + x_base_n
                + c_tile[:, None] * sxc
                + w_ptrs[None, :]
            )
            cmask_w = mask_ct[:, None] & mask_w[None, :]

            # Load 4 heights from the same C/W tile
            v0 = tl.load(base_cw + (h + 0) * sxh, mask=cmask_w, other=INF, cache_modifier=".cg").to(tl.float32)
            v1 = tl.load(base_cw + (h + 1) * sxh, mask=cmask_w, other=INF, cache_modifier=".cg").to(tl.float32)
            v2 = tl.load(base_cw + (h + 2) * sxh, mask=cmask_w, other=INF, cache_modifier=".cg").to(tl.float32)
            v3 = tl.load(base_cw + (h + 3) * sxh, mask=cmask_w, other=INF, cache_modifier=".cg").to(tl.float32)

            # Reduce across C-tile
            min0 = tl.minimum(min0, tl.min(v0, axis=0))
            min1 = tl.minimum(min1, tl.min(v1, axis=0))
            min2 = tl.minimum(min2, tl.min(v2, axis=0))
            min3 = tl.minimum(min3, tl.min(v3, axis=0))

            c_start += BLOCK_C

        # Accumulate valid results into acc with a single mask application
        sum_mins = (min0 + min1) + (min2 + min3)
        acc += tl.where(mask_w, sum_mins, 0.0)
        h += 4

    # Handle remaining rows if H % 4 != 0
    while h < H:
        cur_min = tl.full((BLOCK_W,), INF, dtype=tl.float32)
        c_start = 0
        while c_start < C:
            c_tile = c_start + tl.arange(0, BLOCK_C)
            mask_ct = c_tile < C
            base_cw = (
                x_ptr
                + x_base_n
                + c_tile[:, None] * sxc
                + w_ptrs[None, :]
            )
            x_vals = tl.load(base_cw + h * sxh, mask=mask_ct[:, None] & mask_w[None, :], other=INF, cache_modifier=".cg").to(tl.float32)
            cur_min = tl.minimum(cur_min, tl.min(x_vals, axis=0))
            c_start += BLOCK_C
        acc += tl.where(mask_w, cur_min, 0.0)
        h += 1

    # Apply GELU: 0.5*x*(1+erf(x/sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu_vals = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # Load bias for the channel tile
    bias_vals = tl.load(bias_ptr + c_offsets * sbc, mask=mask_c, other=0.0).to(tl.float32)

    # Write out for this channel tile with bias broadcasting
    out_ptrs = (
        y_ptr
        + out_base_n
        + c_offsets[:, None] * syc
        + w_offsets[None, :] * syw
    )
    out_tile = gelu_vals[None, :] + bias_vals[:, None]
    tl.store(out_ptrs, out_tile, mask=mask_c[:, None] & mask_w[None, :])
