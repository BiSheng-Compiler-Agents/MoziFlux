import torch
import torch.nn as nn
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_NUM_PARTS = 32
_STATS_BLOCK = 1024
_APPLY_BLOCK = 1024


@triton.jit
def _groupnorm_partial_stats_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_ss_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    groups: tl.constexpr,
    channels_per_group: tl.constexpr,
    group_elems: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    part = pid % NUM_PARTS
    group_pid = pid // NUM_PARTS
    n = group_pid // groups
    g = group_pid - n * groups

    c_start = g * channels_per_group
    base = n * C * HW
    offs = tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_SIZE)

    acc = tl.zeros([1], dtype=tl.float32)
    acc2 = tl.zeros([1], dtype=tl.float32)
    start = part * BLOCK_SIZE
    step = NUM_PARTS * BLOCK_SIZE
    for block_start in tl.range(start, group_elems, step):
        idx = block_start + offs
        mask = idx < group_elems
        c_rel = idx // HW
        hw = idx - c_rel * HW
        ptrs = base + (c_start + c_rel) * HW + hw
        x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0, keep_dims=True)
        acc2 += tl.sum(x * x, axis=0, keep_dims=True)

    out = group_pid * NUM_PARTS + part
    one = tl.arange(0, 1)
    tl.store(partial_sum_ptr + out + one, acc, mask=one == 0)
    tl.store(partial_ss_ptr + out + one, acc2, mask=one == 0)


@triton.jit
def _groupnorm_finalize_stats_kernel(
    partial_sum_ptr,
    partial_ss_ptr,
    mean_ptr,
    rstd_ptr,
    total_groups: tl.constexpr,
    group_elems_inv: tl.constexpr,
    eps: tl.constexpr,
    NUM_PARTS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, NUM_PARTS)
    s = tl.load(partial_sum_ptr + pid * NUM_PARTS + offs,
                mask=offs < NUM_PARTS,
                other=0.0).to(tl.float32)
    ss = tl.load(partial_ss_ptr + pid * NUM_PARTS + offs,
                 mask=offs < NUM_PARTS,
                 other=0.0).to(tl.float32)
    mean = tl.sum(s, axis=0) * group_elems_inv
    var = tl.sum(ss, axis=0) * group_elems_inv - mean * mean
    rstd = tl.rsqrt(var + eps)
    tl.store(mean_ptr + pid, mean, mask=pid < total_groups)
    tl.store(rstd_ptr + pid, rstd, mask=pid < total_groups)


@triton.jit
def _groupnorm_apply_persistent_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    gamma_ptr,
    beta_ptr,
    C,
    HW,
    groups,
    channels_per_group,
    hw_tiles,
    total_tiles,
    n_programs: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HW)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_HW)
    for tile_id in tl.range(pid, total_tiles, n_programs):
        hw_tile = tile_id % hw_tiles
        tmp = tile_id // hw_tiles
        c = tmp % C
        n = tmp // C
        idx = hw_tile * BLOCK_HW + offs
        mask = idx < HW
        g = c // channels_per_group
        base = (n * C + c) * HW
        mean = tl.load(mean_ptr + n * groups + g)
        rstd = tl.load(rstd_ptr + n * groups + g)
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = ((x - mean) * rstd) * gamma + beta
        tl.store(y_ptr + base + idx, y, mask=mask)


@triton.jit
def _groupnorm_apply_channel_kernel(
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
    pid = tl.program_id(0)
    n = pid // C
    c = pid - n * C
    g = c // channels_per_group
    base = (n * C + c) * HW
    mean = tl.load(mean_ptr + n * groups + g)
    rstd = tl.load(rstd_ptr + n * groups + g)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta = tl.load(beta_ptr + c).to(tl.float32)

    offs = tl.arange(0, BLOCK_HW)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_HW)
    for start in tl.range(0, HW, BLOCK_HW):
        idx = start + offs
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = ((x - mean) * rstd) * gamma + beta
        tl.store(y_ptr + base + idx, y, mask=mask)


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

    x = x.contiguous()
    N, C = x.shape[:2]
    HW = x.numel() // (N * C)
    x_flat = x.view(N, C, HW)
    assert C % num_groups == 0, "num_groups must divide number of channels"

    channels_per_group = C // num_groups
    group_elems = channels_per_group * HW
    total_groups = N * num_groups
    mean = torch.empty((N, num_groups), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N, num_groups), device=x.device, dtype=torch.float32)
    num_parts = max(1, min(_NUM_PARTS, _MAX_PROGRAMS // max(1, total_groups)))
    partial_sum = torch.empty((total_groups, num_parts),
                              device=x.device,
                              dtype=torch.float32)
    partial_ss = torch.empty((total_groups, num_parts),
                             device=x.device,
                             dtype=torch.float32)
    y_flat = torch.empty_like(x_flat)

    if (weight is None) or (bias is None):
        w = torch.ones(C, device=x.device, dtype=torch.float32)
        b = torch.zeros(C, device=x.device, dtype=torch.float32)
    else:
        w = weight.to(device=x.device, dtype=torch.float32)
        b = bias.to(device=x.device, dtype=torch.float32)

    stats_grid = (total_groups * num_parts, )
    _groupnorm_partial_stats_kernel[stats_grid](
        x_flat,
        partial_sum,
        partial_ss,
        N,
        C,
        HW,
        num_groups,
        channels_per_group,
        group_elems,
        num_parts,
        BLOCK_SIZE=_STATS_BLOCK,
        num_warps=4,
        num_stages=2,
    )
    _groupnorm_finalize_stats_kernel[(total_groups, )](
        partial_sum,
        partial_ss,
        mean,
        rstd,
        total_groups,
        1.0 / float(group_elems),
        float(eps),
        num_parts,
        num_warps=4,
        num_stages=2,
    )

    if N * C <= _MAX_PROGRAMS:
        _groupnorm_apply_channel_kernel[(N * C, )](
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
            BLOCK_HW=_APPLY_BLOCK,
            num_warps=4,
            num_stages=2,
        )
    else:
        hw_tiles = triton.cdiv(HW, _APPLY_BLOCK)
        total_tiles = N * C * hw_tiles
        _groupnorm_apply_persistent_kernel[(_MAX_PROGRAMS, )](
            x_flat,
            y_flat,
            mean,
            rstd,
            w,
            b,
            C,
            HW,
            num_groups,
            channels_per_group,
            hw_tiles,
            total_tiles,
            _MAX_PROGRAMS,
            BLOCK_HW=_APPLY_BLOCK,
            num_warps=4,
            num_stages=2,
        )
    return y_flat.view_as(x)


class ModelNew(nn.Module):
    """Simple model that performs Group Normalization."""

    def __init__(self, num_features: int = 64, num_groups: int = 8):
        super(ModelNew, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups,
                               num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(x, "is_npu", False):
            raise RuntimeError(
                "ModelNew expects an input tensor on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")
        return group_norm_triton(x, self.gn.weight, self.gn.bias,
                                 self.gn.num_groups, self.gn.eps)


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
