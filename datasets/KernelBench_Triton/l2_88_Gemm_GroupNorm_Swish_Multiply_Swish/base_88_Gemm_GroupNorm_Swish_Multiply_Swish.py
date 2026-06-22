import torch
import torch.nn as nn
import triton
import triton.language as tl

FAST_GROUP_SIZE = 32
FAST_BLOCK_M = 48
FAST_NUM_WARPS = 4
FAST_NUM_STAGES = 1
FAST_USE_RSQRT = False
FAST_USE_HINTS = False


@triton.jit
def _fused_gn_swish_mul_swish_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mulw_ptr,
    y_ptr,
    N,
    C,
    G,
    GROUP_SIZE,
    EPS,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    g_idx = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < N
    offs = tl.arange(0, BLOCK_SIZE)
    c_start = g_idx * GROUP_SIZE
    c_idx = c_start + offs
    col_mask = offs < GROUP_SIZE
    mask = row_mask[:, None] & col_mask[None, :]

    base = rows[:, None] * C + c_idx[None, :]
    x = tl.load(x_ptr + base, mask=mask, other=0.0)

    mean = tl.sum(x, axis=1) / GROUP_SIZE
    xc = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / GROUP_SIZE
    inv_std = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(gamma_ptr + c_idx, mask=col_mask, other=0.0)[None, :]
    beta = tl.load(beta_ptr + c_idx, mask=col_mask, other=0.0)[None, :]
    gn = (x - mean[:, None]) * inv_std[:, None]
    y = gn * gamma + beta
    y = y * tl.sigmoid(y)

    mw = tl.load(mulw_ptr + c_idx, mask=col_mask, other=0.0)[None, :]
    y = y * mw
    out = y * tl.sigmoid(y)
    tl.store(y_ptr + base, out, mask=mask)


@triton.jit
def _fused_gn_swish_mul_swish_g32_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mulw_ptr,
    y_ptr,
    N,
    C,
    EPS,
    BLOCK_M: tl.constexpr,
    USE_RSQRT: tl.constexpr,
    USE_HINTS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    g_idx = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < N
    cols = tl.arange(0, 32)
    if USE_HINTS:
        cols = tl.max_contiguous(cols, 32)
    c_idx = g_idx * 32 + cols
    base = rows[:, None] * C + c_idx[None, :]

    x = tl.load(x_ptr + base, mask=row_mask[:, None], other=0.0)
    mean = tl.sum(x, axis=1) * (1.0 / 32.0)
    centered = x - mean[:, None]
    var = tl.sum(centered * centered, axis=1) * (1.0 / 32.0)
    if USE_RSQRT:
        inv_std = tl.rsqrt(var + EPS)
    else:
        inv_std = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(gamma_ptr + c_idx)[None, :]
    beta = tl.load(beta_ptr + c_idx)[None, :]
    y = centered * inv_std[:, None]
    y = y * gamma + beta
    y = y * tl.sigmoid(y)

    mw = tl.load(mulw_ptr + c_idx)[None, :]
    y = y * mw
    out = y * tl.sigmoid(y)
    tl.store(y_ptr + base, out, mask=row_mask[:, None])


def gemm_groupnorm_swish_multiply_swish(
    x,
    linear_weight,
    linear_bias,
    norm_weight,
    norm_bias,
    multiply_weight,
    num_groups,
    eps=1e-5,
):
    if x.dim() != 2:
        raise ValueError("expected `x` to be a 2D tensor")
    if not hasattr(x, "is_npu") or not x.is_npu:
        raise ValueError(
            "the Triton entrypoint only supports Ascend NPU tensors")

    x = x.contiguous()
    linear_weight = linear_weight.contiguous()
    linear_bias = linear_bias.contiguous()
    norm_weight = norm_weight.contiguous()
    norm_bias = norm_bias.contiguous()
    multiply_weight = multiply_weight.contiguous()

    x = torch.matmul(x, linear_weight.transpose(0, 1)) + linear_bias

    n_rows, channels = x.shape
    if channels % num_groups != 0:
        raise ValueError("channels must be divisible by num_groups")

    group_size = channels // num_groups
    y = torch.empty_like(x)

    if group_size == FAST_GROUP_SIZE:
        grid = (triton.cdiv(n_rows, FAST_BLOCK_M), num_groups)
        _fused_gn_swish_mul_swish_g32_kernel[grid](
            x,
            norm_weight,
            norm_bias,
            multiply_weight,
            y,
            n_rows,
            channels,
            eps,
            BLOCK_M=FAST_BLOCK_M,
            USE_RSQRT=FAST_USE_RSQRT,
            USE_HINTS=FAST_USE_HINTS,
            num_warps=FAST_NUM_WARPS,
            num_stages=FAST_NUM_STAGES,
        )
        return y

    block_m = 8
    block_size = 1 << (group_size - 1).bit_length()
    block_size = min(block_size, 1024)
    num_warps = 2 if block_size <= 64 else 4
    grid = (triton.cdiv(n_rows, block_m), num_groups)
    _fused_gn_swish_mul_swish_kernel[grid](
        x,
        norm_weight,
        norm_bias,
        multiply_weight,
        y,
        n_rows,
        channels,
        num_groups,
        group_size,
        eps,
        BLOCK_M=block_m,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, num_groups,
                 multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        return gemm_groupnorm_swish_multiply_swish(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features, )


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]
