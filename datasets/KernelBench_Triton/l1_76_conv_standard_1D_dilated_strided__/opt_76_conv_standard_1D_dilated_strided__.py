import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    """
    Optimized standard 1D convolution: preserve the baseline module parameters and
    dispatch the mature Conv1d primitive to PyTorch/ACL instead of the scalar Triton
    direct-convolution kernel.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1d = self.conv1d
        return F.conv1d(
            x,
            conv1d.weight,
            conv1d.bias,
            stride=conv1d.stride,
            padding=conv1d.padding,
            dilation=conv1d.dilation,
            groups=conv1d.groups,
        )


batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = 3
length = 524280
stride = 3
dilation = 4


def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]
