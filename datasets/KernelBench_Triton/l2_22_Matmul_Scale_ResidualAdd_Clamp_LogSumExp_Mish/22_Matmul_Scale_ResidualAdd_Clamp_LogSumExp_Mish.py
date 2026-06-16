import triton
import triton.language as tl


@triton.jit
def _post_ops_row_lse_mish(
    y_ptr,  # [B, N] input from linear
    out_ptr,  # [B, 1] output stores final x * mish(x)
    B,
    N,  # sizes
    stride_y_m,
    stride_y_n,
    stride_out_m,
    scale_factor,
    clamp_min,
    clamp_max,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    if pid_m >= B:
        return

    off_n = tl.arange(0, BLOCK_N)
    base_ptr = y_ptr + pid_m * stride_y_m

    # Online streaming LogSumExp across columns with numerical stability
    neg_inf = -float("inf")
    m = tl.full((), neg_inf, dtype=tl.float32)  # running max
    s = tl.zeros((), dtype=tl.float32)  # running sum of exp(x - m)

    scale2 = 2.0 * scale_factor

    n = 0
    while n < N:
        n_idx = n + off_n
        mask = n_idx < N

        vals = tl.load(base_ptr + n_idx * stride_y_n,
                       mask=mask,
                       other=0.0,
                       cache_modifier=".cg").to(tl.float32)
        # Fused transform: scale+residual (x *= 2*scale_factor) and clamp
        vals = vals * scale2
        vals = tl.minimum(tl.maximum(vals, clamp_min), clamp_max)

        # Mask OOB lanes to -inf so they don't affect max/sum
        vals = tl.where(mask, vals, neg_inf)

        # Online LSE update
        block_max = tl.max(vals, axis=0)
        m_new = tl.maximum(m, block_max)
        sum_exp_chunk = tl.sum(tl.exp(vals - m_new), axis=0)
        s = s * tl.exp(m - m_new) + sum_exp_chunk
        m = m_new

        n += BLOCK_N

    # lse = m + log(s)
    lse = m + tl.log(s)

    # Compute Mish(lse) = lse * tanh(softplus(lse)), then output = lse * Mish(lse)
    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0) for stability
    abs_lse = tl.abs(lse)
    softplus = tl.log(1.0 + tl.exp(-abs_lse)) + tl.maximum(lse, 0.0)
    # tanh(u) = (1 - exp(-2u)) / (1 + exp(-2u))  (stable for u >= 0)
    eneg2u = tl.exp(-2.0 * softplus)
    tanh_u = (1.0 - eneg2u) / (1.0 + eneg2u)
    # mish = lse * tanh(softplus(lse)); out = lse * mish = lse^2 * tanh(softplus(lse))
    out_val = (lse * lse) * tanh_u

    tl.store(out_ptr + pid_m * stride_out_m, out_val)
