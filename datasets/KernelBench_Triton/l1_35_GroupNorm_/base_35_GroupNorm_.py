import torch
import torch.nn as nn
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
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    USE_HINTS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // groups
    g = pid % groups

    offs_c = tl.arange(0, BLOCK_C)
    valid_c = offs_c < channels_per_group
    c_idx = g * channels_per_group + offs_c
    base = (n * C + c_idx)[:, None] * HW

    sums = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sums_sq = tl.zeros((BLOCK_C,), dtype=tl.float32)

    start = 0
    while start < HW:
        hw = start + tl.arange(0, BLOCK_HW)
        if USE_HINTS:
            hw = tl.max_contiguous(hw, BLOCK_HW)
        mask = valid_c[:, None] & (hw[None, :] < HW)
        x = tl.load(x_ptr + base + hw[None, :], mask=mask, other=0.0).to(tl.float32)
        sums += tl.sum(x, axis=1)
        sums_sq += tl.sum(x * x, axis=1)
        start += BLOCK_HW

    sums = tl.where(valid_c, sums, 0.0)
    sums_sq = tl.where(valid_c, sums_sq, 0.0)
    denom = group_elems.to(tl.float32)
    mean = tl.sum(sums, axis=0) / denom
    var = tl.sum(sums_sq, axis=0) / denom - mean * mean
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
    USE_HINTS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    g = c // channels_per_group
    base = (n * C + c) * HW
    mean = tl.load(mean_ptr + n * groups + g)
    rstd = tl.load(rstd_ptr + n * groups + g)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta = tl.load(beta_ptr + c).to(tl.float32)

    start = 0
    while start < HW:
        offs = start + tl.arange(0, BLOCK_HW)
        if USE_HINTS:
            offs = tl.max_contiguous(offs, BLOCK_HW)
        mask = offs < HW
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = ((x - mean) * rstd) * gamma + beta
        tl.store(y_ptr + base + offs, y, mask=mask)
        start += BLOCK_HW


@triton.jit
def _groupnorm_fused_kernel(
    x_ptr,
    y_ptr,
    gamma_ptr,
    beta_ptr,
    N,
    C,
    HW,
    groups,
    channels_per_group,
    group_elems,
    eps,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    USE_HINTS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // groups
    g = pid % groups

    offs_c = tl.arange(0, BLOCK_C)
    valid_c = offs_c < channels_per_group
    c_idx = g * channels_per_group + offs_c
    base = (n * C + c_idx)[:, None] * HW

    sums = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sums_sq = tl.zeros((BLOCK_C,), dtype=tl.float32)

    start = 0
    while start < HW:
        hw = start + tl.arange(0, BLOCK_HW)
        if USE_HINTS:
            hw = tl.max_contiguous(hw, BLOCK_HW)
        mask = valid_c[:, None] & (hw[None, :] < HW)
        x = tl.load(x_ptr + base + hw[None, :], mask=mask, other=0.0).to(tl.float32)
        sums += tl.sum(x, axis=1)
        sums_sq += tl.sum(x * x, axis=1)
        start += BLOCK_HW

    sums = tl.where(valid_c, sums, 0.0)
    sums_sq = tl.where(valid_c, sums_sq, 0.0)
    denom = group_elems.to(tl.float32)
    mean = tl.sum(sums, axis=0) / denom
    var = tl.sum(sums_sq, axis=0) / denom - mean * mean
    rstd = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + c_idx, mask=valid_c, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + c_idx, mask=valid_c, other=0.0).to(tl.float32)

    start = 0
    while start < HW:
        hw = start + tl.arange(0, BLOCK_HW)
        if USE_HINTS:
            hw = tl.max_contiguous(hw, BLOCK_HW)
        mask = valid_c[:, None] & (hw[None, :] < HW)
        x = tl.load(x_ptr + base + hw[None, :], mask=mask, other=0.0).to(tl.float32)
        y = ((x - mean) * rstd) * gamma[:, None] + beta[:, None]
        tl.store(y_ptr + base + hw[None, :], y, mask=mask)
        start += BLOCK_HW


def group_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
):
    if not getattr(x, "is_npu", False):
        raise RuntimeError("group_norm_triton expects an input tensor on Ascend NPU")
    if x.requires_grad:
        raise RuntimeError("group_norm_triton does not support autograd-enabled inputs")
    if x.ndim < 3:
        raise ValueError("group_norm_triton expects input with shape (N, C, ...)")

    x = x.contiguous()
    N, C = x.shape[:2]
    HW = x.numel() // (N * C)
    x_flat = x.view(N, C, HW)

    if C % num_groups != 0:
        raise ValueError("num_groups must divide number of channels")

    channels_per_group = C // num_groups
    group_elems = channels_per_group * HW
    y_flat = torch.empty_like(x_flat)

    if (weight is None) or (bias is None):
        w = torch.ones(C, device=x.device, dtype=torch.float32)
        b = torch.zeros(C, device=x.device, dtype=torch.float32)
    else:
        w = weight.to(device=x.device, dtype=torch.float32)
        b = bias.to(device=x.device, dtype=torch.float32)

    block_c = 8
    block_hw = 1024
    use_hints = True

    if "fused" == "two_stage":
        mean = torch.empty((N, num_groups), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N, num_groups), device=x.device, dtype=torch.float32)
        _groupnorm_stats_kernel[(N * num_groups,)](
            x_flat,
            mean,
            rstd,
            N,
            C,
            HW,
            num_groups,
            channels_per_group,
            group_elems,
            float(eps),
            BLOCK_C=block_c,
            BLOCK_HW=block_hw,
            USE_HINTS=use_hints,
            num_warps=8,
            num_stages=4,
        )
        _groupnorm_fwd_kernel[(N * C,)](
            x_flat,
            y_flat,
            mean,
            rstd,
            w,
            b,
            N,
            C,
            HW,
            num_groups,
            channels_per_group,
            BLOCK_HW=block_hw,
            USE_HINTS=use_hints,
            num_warps=8,
            num_stages=4,
        )
    else:
        _groupnorm_fused_kernel[(N * num_groups,)](
            x_flat,
            y_flat,
            w,
            b,
            N,
            C,
            HW,
            num_groups,
            channels_per_group,
            group_elems,
            float(eps),
            BLOCK_C=block_c,
            BLOCK_HW=block_hw,
            USE_HINTS=use_hints,
            num_warps=8,
            num_stages=4,
        )
    return y_flat.view_as(x)


class ModelNew(nn.Module):
    def __init__(self, num_features: int = 64, num_groups: int = 8):
        super(ModelNew, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(x, "is_npu", False):
            raise RuntimeError("ModelNew expects an input tensor on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")
        return group_norm_triton(
            x,
            self.gn.weight,
            self.gn.bias,
            self.gn.num_groups,
            self.gn.eps,
        )


batch_size = 112
features = 64
num_groups = 8
dim1 = 512
dim2 = 512


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features, num_groups]
