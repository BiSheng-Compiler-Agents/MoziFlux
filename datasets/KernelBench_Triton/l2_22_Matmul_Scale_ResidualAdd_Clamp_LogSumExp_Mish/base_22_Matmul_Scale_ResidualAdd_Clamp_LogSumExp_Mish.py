import torch
import torch.nn as nn
import torch.nn.functional as F
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    off_n = tl.arange(0, BLOCK_N)
    row_ptrs = y_ptr + rows[:, None] * stride_y_m

    # Online streaming LogSumExp across columns with numerical stability
    neg_inf = -float("inf")
    m = tl.full((BLOCK_M, ), neg_inf, dtype=tl.float32)  # running max
    s = tl.zeros((BLOCK_M, ), dtype=tl.float32)  # running sum of exp(x - m)

    scale2 = 2.0 * scale_factor

    n = 0
    while n < N:
        n_idx = n + off_n
        col_mask = n_idx < N
        mask = row_mask[:, None] & col_mask[None, :]

        vals = tl.load(
            row_ptrs + n_idx[None, :] * stride_y_n,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        # Fused transform: scale+residual (x *= 2*scale_factor) and clamp
        vals = vals * scale2
        vals = tl.minimum(tl.maximum(vals, clamp_min), clamp_max)

        # Mask OOB lanes to -inf so they don't affect max/sum
        vals = tl.where(mask, vals, neg_inf)

        # Online LSE update
        block_max = tl.max(vals, axis=1)
        m_new = tl.maximum(m, block_max)
        sum_exp_chunk = tl.sum(tl.exp(vals - m_new[:, None]), axis=1)
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

    tl.store(out_ptr + rows * stride_out_m, out_val, mask=row_mask)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """

    def __init__(self, input_size, hidden_size, scale_factor, clamp_min,
                 clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        return matmul_scale_residualadd_clamp_logsumexp_mish(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.scale_factor,
            self.clamp_min,
            self.clamp_max,
        )


def _launch_post_ops(y, scale_factor, clamp_min, clamp_max):
    if y.device.type != "npu":
        raise RuntimeError(
            "matmul_scale_residualadd_clamp_logsumexp_mish requires NPU tensors"
        )

    y = y.contiguous()
    B, N = y.shape
    out = torch.empty((B, 1), device=y.device, dtype=y.dtype)

    block_n = 128
    num_warps = 4
    block_m = 16

    grid = (triton.cdiv(B, block_m), )
    _post_ops_row_lse_mish[grid](
        y,
        out,
        B,
        N,
        y.stride(0),
        y.stride(1),
        out.stride(0),
        float(scale_factor),
        float(clamp_min),
        float(clamp_max),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


def matmul_scale_residualadd_clamp_logsumexp_mish(
    x,
    weight,
    bias=None,
    scale_factor=1.0,
    clamp_min=-10.0,
    clamp_max=10.0,
):
    y = F.linear(x, weight, bias)
    return _launch_post_ops(y, scale_factor, clamp_min, clamp_max)


batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]
