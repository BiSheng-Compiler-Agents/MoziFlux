import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized host interface for standard 3D convolution with square input and kernel.

    The baseline launched a Triton no-op/touch kernel over the full input before calling
    torch.nn.Conv3d. That kernel did not contribute to the convolution result; removing it
    preserves semantics and eliminates one full input GM read plus Triton launch overhead.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"expected a 5D input tensor, got shape {tuple(x.shape)}")
        if x.device.type != "npu":
            raise RuntimeError(
                f"ModelNew expects NPU inputs, got device {x.device}")
        return self.conv3d(x)


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, width, height)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
