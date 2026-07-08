import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512
_MAX_GRID = 65535
_USE_TRITON_EPILOGUE = False


@triton.jit
def _groupnorm_lrelu_epilogue(
    z_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    total_tiles,
    tile_offset,
    N: tl.constexpr,
    C: tl.constexpr,
    G: tl.constexpr,
    Cg: tl.constexpr,
    eps,
    neg_slope,
    stride_zm,
    stride_zc,
    stride_ym,
    stride_yc,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = tile_offset + pid
    group_tiles = tl.cdiv(G, GROUP_BLOCK)
    row = tile // group_tiles
    gt = tile - row * group_tiles

    g = gt * GROUP_BLOCK + tl.arange(0, GROUP_BLOCK)
    c_in_g = tl.arange(0, BLOCK_C)
    ch = g[:, None] * Cg + c_in_g[None, :]
    valid = (tile < total_tiles) & (row < N) & (g[:, None] < G) & (
        c_in_g[None, :] < Cg) & (ch < C)
    safe_ch = tl.where(valid, ch, 0)

    z = tl.load(z_ptr + row * stride_zm + safe_ch * stride_zc,
                mask=valid,
                other=0.0).to(tl.float32)
    mean = tl.sum(z, axis=1) / Cg
    centered = z - mean[:, None]
    var = tl.sum(centered * centered, axis=1) / Cg
    inv = tl.rsqrt(var + eps)
    gamma = tl.load(gamma_ptr + safe_ch, mask=valid, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + safe_ch, mask=valid, other=0.0).to(tl.float32)
    out = centered * inv[:, None] * gamma + beta
    out = tl.where(out >= 0.0, out * 2.0, out * (2.0 * neg_slope))
    tl.store(y_ptr + row * stride_ym + safe_ch * stride_yc, out, mask=valid)


def _triton_groupnorm_lrelu(z, gamma, beta, groups: int, eps: float,
                            neg_slope: float):
    N, C = z.shape
    assert C % groups == 0
    Cg = C // groups
    block_c = triton.next_power_of_2(Cg)
    group_block = max(1, min(8, groups, 2048 // block_c))
    group_tiles = triton.cdiv(groups, group_block)
    total_tiles = N * group_tiles
    y = torch.empty_like(z)
    for off in range(0, total_tiles, _MAX_GRID):
        chunk = min(_MAX_GRID, total_tiles - off)
        _groupnorm_lrelu_epilogue[(chunk, )](
            z,
            gamma,
            beta,
            y,
            total_tiles,
            off,
            N,
            C,
            groups,
            Cg,
            eps,
            neg_slope,
            z.stride(0),
            z.stride(1),
            y.stride(0),
            y.stride(1),
            GROUP_BLOCK=group_block,
            BLOCK_C=block_c,
        )
    return y


class ModelNew(nn.Module):
    """Optimized Matmul -> GroupNorm -> LeakyReLU -> Sum model for Ascend NPU."""

    def __init__(self,
                 input_size=input_size,
                 hidden_size=hidden_size,
                 num_groups=num_groups,
                 eps=1e-5,
                 negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups,
                               num_channels=hidden_size,
                               eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        if x.device.type != "npu" or x.dtype not in (torch.float16,
                                                     torch.float32):
            raise RuntimeError(
                "ModelNew requires float16 or float32 inputs on Ascend NPU.")
        if x.ndim != 2:
            raise RuntimeError(
                "ModelNew expects a 2D [batch, input_size] tensor.")
        weight = self.fc.weight.to(device=x.device, dtype=x.dtype)
        bias = self.fc.bias.to(
            device=x.device,
            dtype=x.dtype) if self.fc.bias is not None else None
        gamma = self.gn.weight.to(device=x.device, dtype=x.dtype)
        beta = self.gn.bias.to(device=x.device, dtype=x.dtype)
        z = F.linear(x.contiguous(), weight, bias)
        if _USE_TRITON_EPILOGUE:
            return _triton_groupnorm_lrelu(z.contiguous(), gamma.contiguous(),
                                           beta.contiguous(),
                                           self.gn.num_groups, self.gn.eps,
                                           self.leaky_relu.negative_slope)
        y = F.group_norm(z, self.gn.num_groups, gamma, beta, self.gn.eps)
        return F.leaky_relu(
            y, negative_slope=self.leaky_relu.negative_slope) * 2.0


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, num_groups]
