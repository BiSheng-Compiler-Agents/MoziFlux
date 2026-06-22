import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BATCH_SIZE = 16
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 128
DEFAULT_KERNEL_SIZE = 3
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 1
DEFAULT_DILATION = 1


class ModelNew(nn.Module):
    """Depthwise-separable 2D convolution using Ascend ACL/PyTorch conv kernels."""

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        dilation: int = DEFAULT_DILATION,
        bias: bool = False,
    ):
        super().__init__()
        # Preserve the source module initialization order and parameter ownership.
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.contiguous().to(torch.float32)
        y = F.conv2d(
            x32,
            self.depthwise.weight.to(device=x.device, dtype=torch.float32),
            None if self.depthwise.bias is None else self.depthwise.bias.to(
                device=x.device, dtype=torch.float32),
            stride=self.depthwise.stride,
            padding=self.depthwise.padding,
            dilation=self.depthwise.dilation,
            groups=self.depthwise.groups,
        )
        y = F.conv2d(
            y,
            self.pointwise.weight.to(device=x.device, dtype=torch.float32),
            None if self.pointwise.bias is None else self.pointwise.bias.to(
                device=x.device, dtype=torch.float32),
            stride=self.pointwise.stride,
            padding=self.pointwise.padding,
            dilation=self.pointwise.dilation,
            groups=self.pointwise.groups,
        )
        return y.to(orig_dtype)


batch_size = 16
in_channels = 64
out_channels = 128
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 1
dilation = 1


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
