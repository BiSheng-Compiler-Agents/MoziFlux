import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 128


class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution (1x1 Conv2d) using Ascend ACL/PyTorch dispatch.
    Preserves the baseline ModelNew constructor, parameter ownership, and Conv2d semantics.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects Ascend NPU tensors.")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise RuntimeError(
                f"Unsupported dtype for optimized Conv2d dispatch: {x.dtype}")

        c = self.conv1d
        weight = c.weight if c.weight.dtype == x.dtype else c.weight.to(
            dtype=x.dtype)
        bias = None if c.bias is None else (
            c.bias if c.bias.dtype == x.dtype else c.bias.to(dtype=x.dtype))
        return F.conv2d(
            x,
            weight,
            bias,
            stride=c.stride,
            padding=c.padding,
            dilation=c.dilation,
            groups=c.groups,
        )


batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width, device="npu")
    return [x]


def get_init_inputs():
    return [in_channels, out_channels]
