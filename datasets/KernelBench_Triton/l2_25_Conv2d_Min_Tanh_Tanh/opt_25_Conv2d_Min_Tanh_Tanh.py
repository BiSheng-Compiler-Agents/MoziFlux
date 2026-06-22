import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 16
DEFAULT_OUT_CHANNELS = 64
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_KERNEL_SIZE = 3


def _min_tanh2_acl(x: torch.Tensor) -> torch.Tensor:
    """ACL-backed channel minimum followed by two tanh applications."""
    return torch.tanh(torch.tanh(torch.amin(x, dim=1, keepdim=True)))


class ModelNew(nn.Module):
    """
    Conv2d -> channelwise minimum -> tanh -> tanh.

    The original custom Triton epilogue is replaced with ACL-backed torch reductions
    and activations.  Constructor/parameter semantics are preserved.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew only supports Ascend NPU execution")
        x = F.conv2d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return _min_tanh2_acl(x)


batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
