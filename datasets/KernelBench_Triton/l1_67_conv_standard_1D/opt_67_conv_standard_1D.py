import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    """Optimized standard 1D convolution using Ascend ACL through PyTorch."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            stride=self.conv1d.stride,
            padding=self.conv1d.padding,
            dilation=self.conv1d.dilation,
            groups=self.conv1d.groups,
        )


batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072


def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
