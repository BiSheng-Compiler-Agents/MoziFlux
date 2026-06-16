import triton
import triton.language as tl


@triton.jit
def _groupnorm_stats_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    C,
    HW,
    groups,
    channels_per_group,
    group_elems,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // groups
    g = pid % groups

    c_start = g * channels_per_group
    base = n * C * HW

    offs = tl.arange(0, BLOCK_SIZE)
    s = tl.zeros((), dtype=tl.float32)
    ss = tl.zeros((), dtype=tl.float32)

    start = 0
    while start < group_elems:
        idx = start + offs
        mask = idx < group_elems
        c_rel = idx // HW
        hw = idx - c_rel * HW
        c_idx = c_start + c_rel
        ptrs = base + c_idx * HW + hw
        x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE

    denom = group_elems.to(tl.float32)
    mean = s / denom
    var = ss / denom - mean * mean
    rstd = tl.rsqrt(var + eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def _groupnorm_fwd_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    gamma_ptr,
    beta_ptr,
    N,
    C,
    HW,
    groups,
    channels_per_group,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    g = c // channels_per_group
    base = (n * C + c) * HW
    mean = tl.load(mean_ptr + n * groups + g)
    rstd = tl.load(rstd_ptr + n * groups + g)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    offs = tl.arange(0, BLOCK_HW)
    start = 0
    while start < HW:
        idx = start + offs
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = ((x - mean) * rstd) * gamma + beta
        tl.store(y_ptr + base + idx, y, mask=mask)
        start += BLOCK_HW
