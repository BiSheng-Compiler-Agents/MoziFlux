import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_bias_scale_clamp_inplace(
    in_out_ptr,
    bias_ptr,
    s,
    plane_count,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
    PLANES_PER_PROG: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tiles_per_plane = tl.cdiv(HW, BLOCK_SIZE)
    tile_groups_per_plane = tl.cdiv(tiles_per_plane, TILES_PER_PROG)
    tile_group = pid % tile_groups_per_plane
    plane_group = pid // tile_groups_per_plane
    tile_base = tile_group * TILES_PER_PROG
    upper = tl.minimum(1.0, 1.0 / s)

    for lane in tl.static_range(0, PLANES_PER_PROG):
        plane_id = plane_group * PLANES_PER_PROG + lane
        channel = plane_id % C
        b = tl.load(bias_ptr + channel)
        plane_base = plane_id * HW
        plane_mask = plane_id < plane_count

        for tile_lane in tl.static_range(0, TILES_PER_PROG):
            tile_idx_in_plane = tile_base + tile_lane
            hw_offsets = tile_idx_in_plane * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = plane_mask & (hw_offsets < HW)
            offsets = plane_base + hw_offsets
            x = tl.load(in_out_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
            y = tl.maximum(x + b, 0.0)
            y = tl.minimum(y, upper)

            tl.store(in_out_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        kernel_size=None,
        stride=None,
        padding=None,
        output_padding=None,
        bias_shape=None,
        scaling_factor=None,
    ):
        super(ModelNew, self).__init__()
        if in_channels is None:
            in_channels = 3
        if out_channels is None:
            out_channels = 16
        if kernel_size is None:
            kernel_size = 3
        if stride is None:
            stride = 2
        if padding is None:
            padding = 1
        if output_padding is None:
            output_padding = 1
        if bias_shape is None:
            bias_shape = (out_channels, 1, 1)
        if scaling_factor is None:
            scaling_factor = 2.0
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")

        y = self.conv_transpose(x)
        s = float(self.scaling_factor)
        y = y.contiguous()
        bias = self.bias.to(device=y.device, dtype=y.dtype).contiguous().view(-1)

        _, C, H, W = y.shape
        HW = H * W
        plane_count = y.numel() // HW
        block_size = 4096
        planes_per_prog = 12
        tiles_per_prog = 1
        grid = lambda META: (
            triton.cdiv(plane_count, META["PLANES_PER_PROG"])
            * triton.cdiv(triton.cdiv(HW, META["BLOCK_SIZE"]), META["TILES_PER_PROG"]),
        )
        _fused_bias_scale_clamp_inplace[grid](
            y,
            bias,
            s,
            plane_count,
            C,
            HW,
            BLOCK_SIZE=block_size,
            PLANES_PER_PROG=planes_per_prog,
            TILES_PER_PROG=tiles_per_prog,
            num_warps=2,
            num_stages=1,
        )
        return y


batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
