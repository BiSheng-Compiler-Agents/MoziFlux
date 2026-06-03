import torch
import torch.nn as nn
import triton
import triton.language as tl


DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 8
DEFAULT_OUT_CHANNELS = 64
DEFAULT_DEPTH = 16
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_SUM_TENSOR_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1, 1)


@triton.jit
def _fused_post_conv_kernel(
    x_ptr,
    sum_ptr,
    y_ptr,
    inner,
    nc,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid_inner = tl.program_id(axis=0)
    pid_nc = tl.program_id(axis=1)
    offsets_inner = pid_inner * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets_inner < inner
    base = pid_nc * inner + offsets_inner
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    bias = tl.load(sum_ptr + (pid_nc % C))
    y = tl.where(x >= 0.0, x, x * 0.2)
    y = tl.maximum(tl.minimum(y + bias, 1.0), -1.0)
    gelu = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
    tl.store(y_ptr + base, gelu, mask=mask)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        sum_tensor_shape=DEFAULT_SUM_TENSOR_SHAPE,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        n, c, d, h, w = x.shape
        inner = d * h * w
        x_contig = x.contiguous()
        bias = self.sum_tensor.view(c).contiguous()
        out = torch.empty_like(x_contig)

        nc = n * c
        grid = (triton.cdiv(inner, 2368), nc)
        _fused_post_conv_kernel[grid](
            x_contig,
            bias,
            out,
            inner,
            nc,
            c,
            BLOCK_SIZE=2368
        )

        return out


batch_size = 128
in_channels = 8
out_channels = 64
depth, height, width = 16, 64, 64
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]
