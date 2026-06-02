import triton
import triton.language as tl

@triton.jit
def _groupnorm_hardtanh_kernel(
    x_ptr,           # [N, C] input (post-GEMM), contiguous
    gamma_ptr,       # [C] groupnorm affine weight
    beta_ptr,        # [C] groupnorm affine bias
    out_ptr,         # [N, C] output
    N,               # number of rows (batch size)
    C,               # number of channels (out_features)
    G,               # number of groups
    Cg,              # channels per group = C // G
    eps,             # eps for numerical stability (float)
    minv,            # hardtanh min
    maxv,            # hardtanh max
    BLOCK_SIZE: tl.constexpr,  # power-of-two >= Cg
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    offs = tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, 16)

    offs_c = g * Cg + offs
    row_start = n * C
    base = row_start + offs_c

    ch_mask = offs < Cg
    mask = ch_mask & (n < N)

    # Load input and affine params; compute in fp32 for stability
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_c, mask=ch_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=ch_mask, other=0.0).to(tl.float32)

    # Compute mean and variance via E[x] and E[x^2]
    inv_cg = 1.0 / tl.full((), Cg, tl.float32)
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean = sum_x * inv_cg
    var = sum_x2 * inv_cg - mean * mean

    # Normalize with fused affine: y = x * (gamma * inv_std) + (beta - mean * gamma * inv_std)
    inv_std = tl.rsqrt(var + eps)
    scale = gamma * inv_std
    shift = beta - mean * scale
    y = x * scale + shift

    # HardTanh clamp
    y = tl.maximum(tl.minimum(y, maxv), minv)

    tl.store(out_ptr + base, y, mask=mask)
