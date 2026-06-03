import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _compute_mu_rstd_kernel(
    x_ptr,         # *f32, input tensor after conv, shape [N*C*S]
    m_ptr,         # *f32, multiplier, shape [C]
    mu_ptr,        # *f32, output mean, shape [N*C]
    rstd_ptr,      # *f32, output rstd, shape [N*C]
    S,             # int32, number of spatial elements per (N, C)
    C,             # int32, number of channels
    eps,           # f32, epsilon for numerical stability
    BLOCK_S: tl.constexpr,  # tile over spatial dimension
):
    pid = tl.program_id(axis=0)  # range: [0, N*C)
    n = pid // C
    c = pid % C

    base_nc = n * C + c
    x_base = x_ptr + base_nc * S

    # Load multiplier for this channel
    m = tl.load(m_ptr + c)

    offs = tl.arange(0, BLOCK_S)
    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sq = tl.zeros((), dtype=tl.float32)
    ptrs = x_base + offs

    s = 0
    while s < S:
        mask = s + offs < S
        v = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        v = v * m
        # accumulate scalars to reduce register pressure
        vsum = tl.sum(tl.where(mask, v, 0.0), axis=0)
        vsqsum = tl.sum(tl.where(mask, v * v, 0.0), axis=0)
        acc_sum += vsum
        acc_sq += vsqsum
        s += BLOCK_S
        ptrs += BLOCK_S

    S_f = tl.full((), S, dtype=tl.float32)
    mean = acc_sum / S_f
    var = tl.maximum(0.0, acc_sq / S_f - mean * mean)
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mu_ptr + base_nc, mean)
    tl.store(rstd_ptr + base_nc, rstd)


@triton.jit
def _postprocess_and_reduce_max_kernel(
    x_ptr,         # *f32, input after conv, shape [N*C*S]
    m_ptr,         # *f32, multiplier, shape [C]
    mu_ptr,        # *f32, mean per (N,C), shape [N*C]
    rstd_ptr,      # *f32, rstd per (N,C), shape [N*C]
    out_ptr,       # *f32, output max over C, shape [N*S]
    S,             # int32
    C,             # int32
    clamp_min,     # f32
    clamp_max,     # f32
    BLOCK_S: tl.constexpr,  # tile over spatial
    BLOCK_C: tl.constexpr,  # tile over channels
    BLOCK_SB: tl.constexpr,  # number of spatial tiles per program
):
    pid_n = tl.program_id(axis=0)  # [0, N)
    pid_sb = tl.program_id(axis=1)  # [0, ceil_div(ceil_div(S, BLOCK_S), BLOCK_SB))

    neg_inf = tl.full([BLOCK_S], -1e30, dtype=tl.float32)

    base_nC = pid_n * C
    base_nS = pid_n * S

    for tile_idx in tl.static_range(0, BLOCK_SB):
        sb_index = pid_sb * BLOCK_SB + tile_idx
        s_offs = sb_index * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        max_vals = neg_inf

        c_start = 0
        while c_start < C:
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < C

            # Load per-channel stats and multiplier once per channel tile.
            m_vec = tl.load(m_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
            mu_vec = tl.load(mu_ptr + base_nC + c_offs, mask=c_mask, other=0.0).to(tl.float32)
            rstd_vec = tl.load(rstd_ptr + base_nC + c_offs, mask=c_mask, other=0.0).to(tl.float32)

            nc_idx = (base_nC + c_offs)[:, None]
            ptrs = x_ptr + nc_idx * S + s_offs[None, :]
            mask2d = c_mask[:, None] & s_mask[None, :]

            x_tile = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
            y1 = x_tile * m_vec[:, None]
            normed = (y1 - mu_vec[:, None]) * rstd_vec[:, None]
            normed = tl.maximum(normed, clamp_min)
            normed = tl.minimum(normed, clamp_max)
            y2 = normed * m_vec[:, None]
            y2 = tl.where(mask2d, y2, -1e30)

            cmax = tl.max(y2, axis=0)
            max_vals = tl.maximum(max_vals, cmax)
            c_start += BLOCK_C

        tl.store(out_ptr + base_nS + s_offs, max_vals, mask=s_mask)


class ModelNew(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        multiplier_shape=(16, 1, 1, 1),
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        if x.device.type != "npu":
            raise ValueError("ModelNew expects an Ascend NPU tensor input.")

        x = self.conv(x)

        # Triton fused path:
        # Shapes
        N, C, D, H, W = x.shape
        S = D * H * W

        # Ensure contiguous for predictable indexing
        x = x.contiguous()

        # Flatten multiplier to [C]
        m = self.multiplier.view(C).contiguous()

        # Allocate stats buffers
        mu = torch.empty((N, C), device=x.device, dtype=x.dtype)
        rstd = torch.empty((N, C), device=x.device, dtype=x.dtype)

        # Kernel 1: compute mean and rstd over spatial dims for (x * multiplier)
        grid_mu = (N * C,)
        BLOCK_S1 = 2048
        _compute_mu_rstd_kernel[grid_mu](
            x, m, mu, rstd,
            S, C, self.instance_norm.eps,
            BLOCK_S=BLOCK_S1,
            num_warps=8,
            num_stages=4,
        )

        # Kernel 2: normalize, clamp, second multiply, and reduce max over channels
        out = torch.empty((N, S), device=x.device, dtype=x.dtype)
        BLOCK_S2 = 256
        BLOCK_C2 = 16
        BLOCK_SB2 = 2
        grid_reduce = (N, triton.cdiv(triton.cdiv(S, BLOCK_S2), BLOCK_SB2))
        _postprocess_and_reduce_max_kernel[grid_reduce](
            x, m, mu, rstd, out,
            S, C, float(self.clamp_min), float(self.clamp_max),
            BLOCK_S=BLOCK_S2, BLOCK_C=BLOCK_C2, BLOCK_SB=BLOCK_SB2,
            num_warps=4,
            num_stages=2,
        )

        # Reshape to [N, D, H, W]
        out = out.view(N, D, H, W)
        return out
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]
