import triton
import triton.language as tl

@triton.jit
def _swish_reduce_3d(
    x_ptr,                 # *f32 [N, C, D, H, W]
    sum_ptr,               # *f32 [N * G * D]
    sumsq_ptr,             # *f32 [N * G * D]
    N: tl.constexpr,       # int
    C: tl.constexpr,       # int
    D, H, W,               # int (runtime)
    strideN, strideC, strideD, strideH, strideW,  # int strides
    group_size,            # int
    num_groups,            # int
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program ids
    pid0 = tl.program_id(0)  # over N*C*D
    pid1 = tl.program_id(1)  # over H tiles
    pid2 = tl.program_id(2)  # over W tiles

    # Decode (n, c, d)
    CD = C * D
    n = pid0 // CD
    tmp = pid0 % CD
    c = tmp // D
    d = tmp % D

    # Tile origins
    h_start = pid1 * BLOCK_H
    w_start = pid2 * BLOCK_W

    # Indices within tile
    h_idx = h_start + tl.arange(0, BLOCK_H)[:, None]
    w_idx = w_start + tl.arange(0, BLOCK_W)[None, :]
    mask = (h_idx < H) & (w_idx < W)

    # Offsets for the tile
    offs = (
        n * strideN
        + c * strideC
        + d * strideD
        + h_idx * strideH
        + w_idx * strideW
    )

    # Load and compute Swish
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sigmoid(x) * x  # Swish

    # Partial reductions over tile
    tile_sum_h = tl.sum(s, axis=1)
    tile_sum = tl.sum(tile_sum_h, axis=0)
    ssq = s * s
    tile_sumsq_h = tl.sum(ssq, axis=1)
    tile_sumsq = tl.sum(tile_sumsq_h, axis=0)

    # Accumulate per-(n, g, d) to reduce atomic contention
    g = c // group_size
    idx = (n * num_groups + g) * D + d
    tl.atomic_add(sum_ptr + idx, tile_sum)
    tl.atomic_add(sumsq_ptr + idx, tile_sumsq)

@triton.jit
def _apply_gn_hswish_3d(
    x_ptr,                 # *f32 [N, C, D, H, W]
    mean_ptr,              # *f32 [N * G]
    invstd_ptr,            # *f32 [N * G]
    weight_ptr,            # *f32 [C]
    bias_ptr,              # *f32 [C]
    y_ptr,                 # *f32 [N, C, D, H, W]
    N: tl.constexpr,       # int
    C: tl.constexpr,       # int
    D, H, W,               # int
    strideN, strideC, strideD, strideH, strideW,  # int strides
    group_size,            # int
    num_groups,            # int
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid0 = tl.program_id(0)  # over N*C*D
    pid1 = tl.program_id(1)  # over H tiles
    pid2 = tl.program_id(2)  # over W tiles

    CD = C * D
    n = pid0 // CD
    tmp = pid0 % CD
    c = tmp // D
    d = tmp % D

    h_start = pid1 * BLOCK_H
    w_start = pid2 * BLOCK_W

    h_idx = h_start + tl.arange(0, BLOCK_H)[:, None]
    w_idx = w_start + tl.arange(0, BLOCK_W)[None, :]
    mask = (h_idx < H) & (w_idx < W)

    offs = (
        n * strideN
        + c * strideC
        + d * strideD
        + h_idx * strideH
        + w_idx * strideW
    )

    # Load input and compute Swish again (no large intermediate buffer)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sigmoid(x) * x  # Swish

    g = c // group_size
    stat_idx = n * num_groups + g
    mu = tl.load(mean_ptr + stat_idx)
    invstd = tl.load(invstd_ptr + stat_idx)
    gamma = tl.load(weight_ptr + c)
    beta = tl.load(bias_ptr + c)

    # GroupNorm affine: ((s - mu) * invstd) * gamma + beta
    v = ((s - mu) * invstd) * gamma + beta

    # HardSwish: v * clamp(v + 3, 0, 6) / 6
    vp3 = v + 3.0
    clamp6 = tl.minimum(tl.maximum(vp3, 0.0), 6.0)
    hs = v * clamp6 * (1.0 / 6.0)

    tl.store(y_ptr + offs, hs, mask=mask)
