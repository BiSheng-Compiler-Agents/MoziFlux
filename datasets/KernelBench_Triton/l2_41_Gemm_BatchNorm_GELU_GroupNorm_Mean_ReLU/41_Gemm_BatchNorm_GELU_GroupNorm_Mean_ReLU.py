import triton
import triton.language as tl

@triton.jit
def _fused_gelu_groupnorm_mean_relu(
    x_ptr,               # [N, C]
    weight_ptr,          # [C]
    bias_ptr,            # [C]
    out_ptr,             # [N, 1]
    C,                   # int: number of features (channels)
    GROUP_SIZE,          # int: channels per group = C // NUM_GROUPS
    EPS: tl.constexpr,   # groupnorm eps (compile-time)
    NUM_GROUPS: tl.constexpr,  # number of groups (compile-time)
    BLOCK_SIZE: tl.constexpr,  # equals GROUP_SIZE (compile-time)
):
    pid = tl.program_id(axis=0)  # row id
    row_start = pid * C

    # Accumulator for sum over features after GroupNorm (fp32)
    total = tl.zeros((), dtype=tl.float32)

    offs = tl.arange(0, BLOCK_SIZE)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    gs = tl.full((), GROUP_SIZE, tl.float32)

    # Provide alignment hints to the compiler
    tl.multiple_of(offs, 16)

    # Iterate groups; NUM_GROUPS is constexpr so loop is unrolled
    for g in tl.static_range(NUM_GROUPS):
        c_start = g * GROUP_SIZE
        cols = c_start + offs
        mask = offs < GROUP_SIZE

        # Load x for this group, apply exact GELU (compute in fp32)
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        xg = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

        # Compute group statistics: mean and variance of GELU(x)
        sum_x = tl.sum(xg, axis=0)
        sum_x2 = tl.sum(xg * xg, axis=0)
        mu = sum_x / gs
        var = sum_x2 / gs - mu * mu
        rstd = tl.rsqrt(var + EPS)

        # Load affine parameters (prefer caching as they are reused across rows)
        gamma = tl.load(weight_ptr + cols, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
        beta = tl.load(bias_ptr + cols, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)

        # Closed-form sum over GroupNorm+affine without materializing y
        # sum(y) = rstd * (sum(gamma * xg) - mu * sum(gamma)) + sum(beta)
        sum_gamma_x = tl.sum(gamma * xg, axis=0)
        sum_gamma = tl.sum(gamma, axis=0)
        sum_beta = tl.sum(beta, axis=0)

        contrib = (sum_gamma_x - mu * sum_gamma) * rstd + sum_beta
        total += contrib

    # Mean over all features then ReLU
    invC = 1.0 / C
    mean_row = total * invC
    out_val = tl.maximum(mean_row, 0.0)
    tl.store(out_ptr + pid, out_val)
