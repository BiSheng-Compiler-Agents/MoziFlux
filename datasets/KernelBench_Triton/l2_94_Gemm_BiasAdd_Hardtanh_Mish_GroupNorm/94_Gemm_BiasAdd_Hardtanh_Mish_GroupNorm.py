import triton
import triton.language as tl

@triton.jit
def fused_bias_act_gn_kernel(
    x_ptr,            # [N, C]
    extra_bias_ptr,   # [C]
    gamma_ptr,        # [C] groupnorm weight
    beta_ptr,         # [C] groupnorm bias
    out_ptr,          # [N, C]
    N, C, G,          # ints
    GROUP_SIZE,       # int = C // G
    eps,              # float
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G

    offs = tl.arange(0, BLOCK_SIZE)
    base = n * C
    g_off = g * GROUP_SIZE
    c = g_off + offs
    mask_c = offs < GROUP_SIZE
    mask = mask_c & (n < N)
    idx = base + c

    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    b = tl.load(extra_bias_ptr + c, mask=mask_c, other=0.0)
    v = x + b
    v = tl.minimum(tl.maximum(v, -1.0), 1.0)

    vf = v.to(tl.float32)
    abs_v = tl.abs(vf)
    zero = tl.zeros_like(vf)
    one = zero + 1.0
    twenty = zero + 20.0
    neg_twenty = zero - 20.0
    softplus_mid = tl.where(vf > zero, vf, zero) + tl.log(one + tl.exp(-abs_v))
    softplus = tl.where(vf > twenty, vf, tl.where(vf < neg_twenty, tl.exp(vf), softplus_mid))
    mish = vf * tl.tanh(softplus)

    mish_masked = tl.where(mask, mish, 0.0)
    sum1 = tl.sum(mish_masked, axis=0)
    sum2 = tl.sum(mish_masked * mish_masked, axis=0)
    gs = tl.full((), GROUP_SIZE, dtype=tl.float32)
    mean = sum1 / gs
    var = sum2 / gs - mean * mean
    inv_std = tl.rsqrt(var + eps)

    y = (mish - mean) * inv_std

    gamma = tl.load(gamma_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
    y = y * gamma + beta

    tl.store(out_ptr + idx, y.to(x.dtype), mask=mask)
