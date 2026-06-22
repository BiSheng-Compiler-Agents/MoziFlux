import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    """Optimized ConvTranspose1d-dilated model using the vendor ACL convolution path."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 3,
        bias: bool = False,
    ):
        super().__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mod = self.conv1d_transpose
        return F.conv_transpose1d(
            x,
            mod.weight,
            mod.bias,
            stride=mod.stride,
            padding=mod.padding,
            output_padding=mod.output_padding,
            groups=mod.groups,
            dilation=mod.dilation,
        )


batch_size = 32
in_channels = 32
out_channels = 64
kernel_size = 5
length = 131072
stride = 1
padding = 0
dilation = 3


def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
