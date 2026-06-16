import triton
import triton.language as tl


@triton.jit
def _touch_tensor_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    _ = tl.load(x_ptr + offs, mask=mask, other=0.0)


@triton.jit
def _permute_flip_weight_kernel(
    src_ptr,  # (in_c, out_c_per_group, kH, kW)
    dst_ptr,  # (out_c, in_c_per_group, kH, kW)
    s_wi,
    s_wo,
    s_wh,
    s_ww,
    s_do,
    s_di,
    s_dh,
    s_dw,
    in_c,
    out_c,
    kH,
    kW,
    groups,
    num_kw_tiles,
    BLOCK_KW: tl.constexpr,
):
    # Program IDs
    pid_o = tl.program_id(0)  # out channel (total)
    pid_i = tl.program_id(1)  # in channel within group
    pid_t = tl.program_id(2)  # fused (kh, kw-tile)

    # Decompose fused pid into kh and kw-tile
    tile_idx = pid_t % num_kw_tiles
    kh_idx = pid_t // num_kw_tiles

    out_per_group = out_c // groups
    in_per_group = in_c // groups

    # Compute group for this output channel
    g = pid_o // out_per_group
    o_within = pid_o % out_per_group
    i_total = g * in_per_group + pid_i

    # Offsets along kW
    kw_offsets = tile_idx * BLOCK_KW + tl.arange(0, BLOCK_KW)
    tl.max_contiguous(kw_offsets, BLOCK_KW)
    tl.multiple_of(kw_offsets, 8)
    mask_kw = kw_offsets < kW
    kw_flipped = (kW - 1) - kw_offsets

    # Source pointer (flip along h and w)
    src_ptrs = (src_ptr + i_total * s_wi + o_within * s_wo +
                (kH - 1 - kh_idx) * s_wh + kw_flipped * s_ww)
    vals = tl.load(src_ptrs, mask=mask_kw, other=0, cache_modifier=".cg")

    # Destination pointer (permute to (out_c, in_c_per_group, kH, kW))
    dst_ptrs = (dst_ptr + pid_o * s_do + pid_i * s_di + kh_idx * s_dh +
                kw_offsets * s_dw)
    tl.store(dst_ptrs, vals, mask=mask_kw)
