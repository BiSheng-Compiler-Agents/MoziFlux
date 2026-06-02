import triton
import triton.language as tl

@triton.jit
def _swish_bias_groupnorm_kernel(
    X_ptr,          # [B, C] post-matmul tensor
    EXTRA_BIAS_ptr, # [C] extra bias added before GroupNorm
    GAMMA_ptr,      # [C] GroupNorm weight
    BETA_ptr,       # [C] GroupNorm bias
    Y_ptr,          # [B, C] output
    B,              # int: batch size
    C,              # int: num channels
    G,              # int: num groups
    EPS,            # float: epsilon
    BLOCK_SIZE: tl.constexpr,  # tile size over channels per group
):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G
    group_size = C // G

    # channel offsets within this group
    offs = tl.arange(0, BLOCK_SIZE)
    ch_start = g * group_size
    ch_idx = ch_start + offs
    in_group = offs < group_size
    n_in = n < B
    mask = in_group & n_in

    # row offset for batch element n
    row_off = n * C
    x_ptrs = X_ptr + row_off + ch_idx
    y_ptrs = Y_ptr + row_off + ch_idx

    # Load input, apply Swish, add extra bias
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_swish = x * tl.sigmoid(x)
    extra_b = tl.load(EXTRA_BIAS_ptr + ch_idx, mask=in_group, other=0.0).to(tl.float32)
    y = x_swish + extra_b

    # Compute mean and variance using E[y^2] - E[y]^2 over the group
    y_grp = tl.where(in_group, y, 0.0)
    sum_y = tl.sum(y_grp, axis=0)
    sum_y2 = tl.sum(y_grp * y_grp, axis=0)
    inv_gs = 1.0 / tl.full((), group_size, tl.float32)
    mean = sum_y * inv_gs
    var = sum_y2 * inv_gs - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = tl.rsqrt(var + EPS)

    # Load affine parameters
    gamma = tl.load(GAMMA_ptr + ch_idx, mask=in_group, other=1.0).to(tl.float32)
    beta = tl.load(BETA_ptr + ch_idx, mask=in_group, other=0.0).to(tl.float32)

    # Fuse scale/shift to reduce ops: out = y * (gamma*inv_std) + (beta - mean*(gamma*inv_std))
    scale = gamma * inv_std
    shift = beta - mean * scale
    out = y * scale + shift

    tl.store(y_ptrs, out, mask=mask)
