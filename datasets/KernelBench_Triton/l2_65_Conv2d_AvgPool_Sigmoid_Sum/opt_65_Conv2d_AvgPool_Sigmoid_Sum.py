import torch
import torch_npu  # noqa: F401
import torch.nn as nn


class ModelNew(nn.Module):
    """
    Optimized host dispatch for Conv2d -> AvgPool2d -> Sigmoid -> Sum.

    The input Triton implementation kept convolution on ACL but replaced standard
    pooling/activation/reduction with two custom Triton launches.  For this standard
    chain the optimized path keeps every post-conv operation on ACL/CANN, removing
    the custom scalar-heavy NCHW pooling kernel and the second channel-sum launch.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        y = self.conv(x)
        y = self.avg_pool(y)
        y = torch.sigmoid(y)
        return torch.sum(y, dim=(1, 2, 3))


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
