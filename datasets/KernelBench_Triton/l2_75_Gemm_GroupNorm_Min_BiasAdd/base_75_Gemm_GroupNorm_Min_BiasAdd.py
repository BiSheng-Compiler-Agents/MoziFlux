import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_groupnorm_min_bias_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    STRIDE_XN,
    STRIDE_OC,
    STRIDE_ON,
    EPS,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    row_base = pid * STRIDE_XN
    offs_g = tl.arange(0, GROUP_SIZE)[None, :]
    row_min = tl.full((), float("inf"), tl.float32)

    num_group_tiles = tl.cdiv(NUM_GROUPS, BLOCK_GROUPS)
    for tile_idx in range(num_group_tiles):
        group_start = tile_idx * BLOCK_GROUPS
        offs_grp = group_start + tl.arange(0, BLOCK_GROUPS)[:, None]
        valid_groups = offs_grp < NUM_GROUPS
        ch_offs_2d = offs_grp * GROUP_SIZE + offs_g
        valid_channels = ch_offs_2d < C
        mask = valid_groups & valid_channels

        x = tl.load(x_ptr + row_base + ch_offs_2d, mask=mask,
                    other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + ch_offs_2d, mask=mask,
                        other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + ch_offs_2d, mask=mask,
                       other=0.0).to(tl.float32)

        inv_gs = 1.0 / GROUP_SIZE
        sum1 = tl.sum(x, axis=1)
        sum2 = tl.sum(x * x, axis=1)
        mean = sum1 * inv_gs
        var = sum2 * inv_gs - mean * mean
        inv_std = tl.rsqrt(var + EPS)
        y = (x - mean[:, None]) * inv_std[:, None]
        y = y * gamma + beta

        gmin = tl.min(tl.where(mask, y, float("inf")), axis=1)
        row_min = tl.minimum(row_min, tl.min(gmin, axis=0))

    for tile_idx in range(num_group_tiles):
        group_start = tile_idx * BLOCK_GROUPS
        offs_grp = group_start + tl.arange(0, BLOCK_GROUPS)[:, None]
        valid_groups = offs_grp < NUM_GROUPS
        ch_offs_2d = offs_grp * GROUP_SIZE + offs_g
        valid_channels = ch_offs_2d < C
        mask = valid_groups & valid_channels
        bias = tl.load(bias_ptr + ch_offs_2d, mask=mask,
                       other=0.0).to(tl.float32)
        out_tile = bias + row_min
        out_ptrs = out_ptr + ch_offs_2d * STRIDE_OC + pid * STRIDE_ON
        tl.store(out_ptrs, out_tile, mask=mask)


def _groupnorm_min_bias_triton(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
):
    if x.ndim != 2:
        raise ValueError(
            f"Expected a 2D input tensor [N, C], got shape {tuple(x.shape)}")
    if x.device.type != "npu":
        raise RuntimeError(
            "The Triton fused GroupNorm-Min-Bias operator requires Ascend NPU tensors."
        )

    N, C = x.shape
    if C % num_groups != 0:
        raise ValueError(
            f"Channel count {C} must be divisible by num_groups={num_groups}")
    if gamma.numel() != C or beta.numel() != C:
        raise ValueError("GroupNorm affine parameters must have shape [C].")

    bias = bias.reshape(-1)
    if bias.numel() != C:
        raise ValueError("Bias must contain exactly C elements.")

    for name, tensor in {"gamma": gamma, "beta": beta, "bias": bias}.items():
        if tensor.device != x.device:
            raise RuntimeError(f"{name} must be on the same device as x.")

    group_size = C // num_groups
    x_ctg = x.contiguous()
    gamma_ctg = gamma.contiguous()
    beta_ctg = beta.contiguous()
    bvec = bias.contiguous()
    out = torch.empty((1, C, N, 1), device=x.device, dtype=x.dtype)

    block_groups = 32 if num_groups >= 32 else num_groups
    _fused_groupnorm_min_bias_kernel[(N, )](
        x_ctg,
        gamma_ctg,
        beta_ctg,
        bvec,
        out,
        N,
        C,
        x_ctg.stride(0),
        out.stride(1),
        out.stride(2),
        eps,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups,
        BLOCK_GROUPS=block_groups,
    )
    return out


def gemm_groupnorm_min_bias_add(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
):
    gemm_out = F.linear(x, linear_weight, linear_bias)
    return _groupnorm_min_bias_triton(
        gemm_out,
        group_norm_weight,
        group_norm_bias,
        bias,
        num_groups,
        eps,
    )


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return gemm_groupnorm_min_bias_add(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
