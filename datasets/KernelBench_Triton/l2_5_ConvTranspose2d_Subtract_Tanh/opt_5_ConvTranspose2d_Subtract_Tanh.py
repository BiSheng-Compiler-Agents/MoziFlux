import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.language.math import tanh as tl_tanh


_BLOCK_HW = 4096
_MAX_PROGRAMS = 65535  # Ascend FFTS grid cap


@triton.jit
def _bias_sub_tanh_plane_direct(
    x_ptr,
    b_ptr,
    y_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    NUM_HW_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    tile_id = tl.program_id(0)
    hw_tile = tile_id % NUM_HW_TILES
    plane = tile_id // NUM_HW_TILES
    c_idx = plane % C

    offs_hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = (tile_id < TOTAL_TILES) & (offs_hw < HW)
    offsets = plane.to(tl.int64) * HW + offs_hw.to(tl.int64)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    b = tl.load(b_ptr + c_idx, mask=tile_id < TOTAL_TILES, other=0.0)
    y = tl_tanh(x.to(tl.float32) - b.to(tl.float32)).to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _bias_sub_tanh_plane_persistent(
    x_ptr,
    b_ptr,
    y_ptr,
    n_programs,
    HW: tl.constexpr,
    C: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    NUM_HW_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    tile_id = tl.program_id(0)
    while tile_id < TOTAL_TILES:
        hw_tile = tile_id % NUM_HW_TILES
        plane = tile_id // NUM_HW_TILES
        c_idx = plane % C

        offs_hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = offs_hw < HW
        offsets = plane.to(tl.int64) * HW + offs_hw.to(tl.int64)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        b = tl.load(b_ptr + c_idx)
        y = tl_tanh(x.to(tl.float32) - b.to(tl.float32)).to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)
        tile_id += n_programs


def _bias_sub_tanh_fused(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("The fused Triton kernel only supports Ascend NPU tensors.")
    x = x.contiguous()
    b = bias.reshape(-1).to(device=x.device, dtype=x.dtype).contiguous()
    y = torch.empty_like(x)

    N, C, H, W = x.shape
    HW = H * W
    n_planes = N * C
    num_hw_tiles = triton.cdiv(HW, _BLOCK_HW)
    total_tiles = n_planes * num_hw_tiles

    if total_tiles > _MAX_PROGRAMS:
        grid = (_MAX_PROGRAMS,)
        _bias_sub_tanh_plane_persistent[grid](
            x, b, y, _MAX_PROGRAMS,
            HW, C, total_tiles, num_hw_tiles,
            BLOCK_HW=_BLOCK_HW,
            num_warps=8,
            num_stages=2,
        )
    else:
        grid = (total_tiles,)
        _bias_sub_tanh_plane_direct[grid](
            x, b, y,
            HW, C, total_tiles, num_hw_tiles,
            BLOCK_HW=_BLOCK_HW,
            num_warps=8,
            num_stages=2,
        )
    return y


def conv_transpose2d_subtract_tanh(
    x: torch.Tensor,
    weight: torch.Tensor,
    subtract_bias: torch.Tensor,
    conv_bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 2,
    padding: int | tuple[int, int] = 1,
    output_padding: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    if x.device.type != "npu" or weight.device.type != "npu" or subtract_bias.device.type != "npu":
        raise RuntimeError("conv_transpose2d_subtract_tanh expects NPU tensors.")
    if conv_bias is not None and conv_bias.device.type != "npu":
        raise RuntimeError("conv_transpose2d_subtract_tanh expects conv_bias on NPU.")

    x = F.conv_transpose2d(
        x,
        weight,
        bias=conv_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    return _bias_sub_tanh_fused(x, subtract_bias)


class ModelNew(nn.Module):
    """Model that performs transposed convolution, subtracts channel bias, and applies tanh."""

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape,
                 stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return conv_transpose2d_subtract_tanh(
            x,
            self.conv_transpose.weight,
            self.bias,
            conv_bias=self.conv_transpose.bias,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            dilation=self.conv_transpose.dilation,
            groups=self.conv_transpose.groups,
        )


batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
