import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_gn_swish_mul_swish_kernel(
    x_ptr,  # (N, C)
    gamma_ptr,  # (C,)
    beta_ptr,  # (C,)
    mulw_ptr,  # (C,)
    y_ptr,  # (N, C)
    N,  # batch size
    C,  # out_features / channels
    G,  # num_groups
    GROUP_SIZE,  # C // G
    EPS,  # eps for GroupNorm
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
        raise ValueError("the Triton entrypoint only supports Ascend NPU tensors")

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
    block_size = 1 << (group_size - 1).bit_length()
    block_size = min(block_size, 1024)
    num_warps = 2 if block_size <= 64 else 4

    y = torch.empty_like(x)
    grid = (n_rows * num_groups,)
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
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    Fused Triton kernel implements: GroupNorm + Swish + Multiply + Swish.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
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
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]
def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]