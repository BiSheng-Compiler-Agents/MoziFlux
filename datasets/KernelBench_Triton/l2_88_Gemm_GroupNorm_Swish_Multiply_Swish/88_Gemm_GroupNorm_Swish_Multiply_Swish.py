import triton
import triton.language as tl

@triton.jit
def _fused_gn_swish_mul_swish_kernel(
    x_ptr,            # (N, C)
    gamma_ptr,        # (C,)
    beta_ptr,         # (C,)
    mulw_ptr,         # (C,)
    y_ptr,            # (N, C)
    N,                # batch size
    C,                # out_features / channels
    G,                # num_groups
    GROUP_SIZE,       # C // G
    EPS,              # eps for GroupNorm
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // G
    g_idx = pid % G

    # Channel indices for this group
    offs = tl.arange(0, BLOCK_SIZE)
    c_start = g_idx * GROUP_SIZE
    c_idx = c_start + offs
    mask = offs < GROUP_SIZE

    # Base offset for this sample
    base = b_idx * C

    # Load group slice once
    x = tl.load(x_ptr + base + c_idx, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / GROUP_SIZE

    # Compute variance using centered values
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / GROUP_SIZE
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Normalize + affine
    gamma = tl.load(gamma_ptr + c_idx, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + c_idx, mask=mask, other=0.0)
    gn = (x - mean) * inv_std
    y = gn * gamma + beta

    # Swish: y * sigmoid(y)
    y = y * tl.sigmoid(y)

    # Multiply with external weight
    mw = tl.load(mulw_ptr + c_idx, mask=mask, other=0.0)
    y = y * mw

    # Second Swish
    out = y * tl.sigmoid(y)

    tl.store(y_ptr + base + c_idx, out, mask=mask)
