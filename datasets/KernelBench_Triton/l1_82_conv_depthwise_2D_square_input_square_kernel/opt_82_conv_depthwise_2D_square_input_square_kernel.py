import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_IN_CHANNELS = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 0


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution for square input / square kernel.

    The baseline expressed Conv2d as a custom Triton vector kernel with one program per
    (N, C, output-row, W-tile). This operator is a mature ACL primitive on Ascend, so the
    optimized path preserves the same parameters and dispatches directly to torch/ACL
    grouped convolution instead of launching the scalar/vector Triton implementation.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups,
        )


batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
