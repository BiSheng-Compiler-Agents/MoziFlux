import triton
import triton.language as tl


@triton.jit
def fused_linear_groupnorm_lrelu_double(
    x_ptr,  # float32 [N, K]
    w_ptr,  # float32 [C, K]
    b_ptr,  # float32 [C]
    gamma_ptr,  # float32 [C]
    beta_ptr,  # float32 [C]
    y_ptr,  # float32 [N, C]
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # hidden size (channels)
    K: tl.constexpr,  # input size
    groups: tl.constexpr,  # number of groups
    eps,  # float32
    neg_slope,  # float32
    stride_xm,
    stride_xk,
    stride_wc,
    stride_wk,
    stride_ym,
    stride_yc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # set at launch to group_size = C // groups
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # tile along batch dimension
    pid_g = tl.program_id(axis=1)  # group id

    # Offsets along M and within the group's channels
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    group_size = BLOCK_N
    c0 = pid_g * group_size
    offs_n = tl.arange(0, BLOCK_N)
    c_idx = c0 + offs_n
    mask_n = c_idx < C

    # Accumulator for matmul: [BM, BN]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop: stream from global with L2-only cache (.cg)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # X tile [BM, BK]
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm +
                          offs_k[None, :] * stride_xk)
        x_tile = tl.load(x_ptrs,
                         mask=(mask_m[:, None] & mask_k[None, :]),
                         other=0.0,
                         cache_modifier=".cg")

        # W tile [BK, BN] from layout [C, K]
        w_ptrs = w_ptr + (offs_k[:, None] * stride_wk +
                          c_idx[None, :] * stride_wc)
        w_tile = tl.load(w_ptrs,
                         mask=(mask_k[:, None] & mask_n[None, :]),
                         other=0.0,
                         cache_modifier=".cg")

        acc += tl.dot(x_tile, w_tile)

    # Add bias per-channel
    b = tl.load(b_ptr + c_idx, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # GroupNorm across channels in the group per sample (row in M)
    # Use E[x^2] - E[x]^2 formulation for fewer FLOPs
    acc_masked = tl.where(mask_n[None, :], acc, 0.0)
    sum_vals = tl.sum(acc_masked, axis=1)
    mean = sum_vals / group_size

    sq = acc_masked * acc_masked
    sum_sq = tl.sum(sq, axis=1)
    ex2 = sum_sq / group_size
    var = tl.maximum(ex2 - mean * mean, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize
    norm = (acc_masked - mean[:, None]) * inv_std[:, None]

    # Affine
    gamma = tl.load(gamma_ptr + c_idx, mask=mask_n, other=1.0)
    beta = tl.load(beta_ptr + c_idx, mask=mask_n, other=0.0)
    out = norm * gamma[None, :] + beta[None, :]

    # Fused LeakyReLU + doubling (x + x)
    pos_scale = 2.0
    neg_scale = 2.0 * neg_slope
    out = tl.where(out >= 0, out * pos_scale, out * neg_scale)

    # Store
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + c_idx[None, :] * stride_yc)
    tl.store(y_ptrs, out, mask=(mask_m[:, None] & mask_n[None, :]))
