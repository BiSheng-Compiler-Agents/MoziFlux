import torch
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    """
    Optimized model: preserve the ConvTranspose2d module/parameters and dispatch
    GELU + GroupNorm to ACL/PyTorch fused library implementations instead of the
    custom two-pass Triton GroupNorm kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups,
                 num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups,
                                       num_channels=out_channels)
        self.groups = groups

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects NPU inputs")
        y = self.conv_transpose(x)
        y = F.gelu(y, approximate="none")
        return F.group_norm(
            y,
            self.group_norm.num_groups,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.eps,
        )


batch_size = 128
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 3
stride = 1
groups = 8
num_groups = 8


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
