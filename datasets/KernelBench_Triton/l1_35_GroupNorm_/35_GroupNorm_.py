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


def group_norm_triton(x: torch.Tensor, weight: torch.Tensor,
                      bias: torch.Tensor, num_groups: int, eps: float):
    if not getattr(x, "is_npu", False):
        raise RuntimeError(
            "group_norm_triton expects an input tensor on Ascend NPU")
    if x.requires_grad:
        raise RuntimeError(
            "group_norm_triton does not support autograd-enabled inputs")
    if x.ndim < 3:
        raise ValueError(
            "group_norm_triton expects input with shape (N, C, ...)")

    # Ensure contiguous layout and flatten spatial dims
    x = x.contiguous()
    N, C = x.shape[:2]
    HW = x.numel() // (N * C)
    x_flat = x.view(N, C, HW)

    assert C % num_groups == 0, "num_groups must divide number of channels"
    channels_per_group = C // num_groups
    group_elems = channels_per_group * HW
    mean = torch.empty((N, num_groups), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N, num_groups), device=x.device, dtype=torch.float32)
    y_flat = torch.empty_like(x_flat, dtype=torch.float32)

    # Prepare affine params in fp32
    if (weight is None) or (bias is None):
        w = torch.ones(C, device=x.device, dtype=torch.float32)
        b = torch.zeros(C, device=x.device, dtype=torch.float32)
    else:
        w = weight.to(device=x.device, dtype=torch.float32)
        b = bias.to(device=x.device, dtype=torch.float32)

    stats_grid = (N * num_groups, )
    _groupnorm_stats_kernel[stats_grid](
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
        BLOCK_SIZE=1024,
        num_warps=4,
        num_stages=2,
    )

    apply_grid = (N * C, )
    _groupnorm_fwd_kernel[apply_grid](
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
        BLOCK_HW=256,
        num_warps=4,
        num_stages=2,
    )
    y = y_flat.view_as(x).to(x.dtype)
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs Group Normalization.
    """

    def __init__(self, num_features: int = 64, num_groups: int = 8):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups,
                               num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        if not getattr(x, "is_npu", False):
            raise RuntimeError(
                "ModelNew expects an input tensor on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")
        return group_norm_triton(x, self.gn.weight, self.gn.bias,
                                 self.gn.num_groups, self.gn.eps)


batch_size = 112  # scaled up
features = 64
num_groups = 8
dim1 = 512
dim2 = 512


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features, num_groups]  # num_features
