import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 16
DEFAULT_IN_CHANNELS = 32
DEFAULT_OUT_CHANNELS = 64
DEFAULT_DEPTH = 16
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 1
DEFAULT_OUTPUT_PADDING = 1
DEFAULT_BIAS_SHAPE = None


@triton.jit
def _fused_bias_residual_mul_add_3d(
    x_ptr,  # pointer to conv_transpose output, shape [N, C, D, H, W] flattened
    bias_ptr,  # pointer to bias per channel, shape [C]
    out_ptr,  # pointer to output tensor, same shape as x_ptr
    DHW,  # int: D*H*W
    C,  # int: number of channels
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_nc = tl.program_id(axis=0)  # over N*C groups
    pid_k = tl.program_id(axis=1)  # tiles along DHW

    # Offsets within DHW
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < DHW

    # Base offset for this (n, c) group
    base = pid_nc * DHW

    # Load x and create original_x (detached in PyTorch, numerically same as x)
    x_val = tl.load(x_ptr + base + offs_k, mask=mask, other=0.0)
    original_x = x_val  # numerically identical to clone().detach()

    # Load per-channel bias
    c_idx = pid_nc % C
    b = tl.load(bias_ptr + c_idx)

    # Replicate the exact operation ordering:
    # x = x + bias
    x_val = x_val + b
    # x = x + original_x
    x_val = x_val + original_x
    # x = x * original_x
    x_val = x_val * original_x
    # x = x + original_x
    x_val = x_val + original_x

    # Store result
    tl.store(out_ptr + base + offs_k, x_val, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum,
    a residual add, a multiplication, and another residual add.
    """

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
        output_padding=DEFAULT_OUTPUT_PADDING,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super(ModelNew, self).__init__()
        if bias_shape is None:
            bias_shape = (out_channels, 1, 1, 1)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Compute conv transpose first
        x = self.conv_transpose(x)
        if not x.is_npu:
            raise RuntimeError(
                "ModelNew requires Ascend NPU tensors for the Triton path")

        # Triton fused pass for: +bias, +original_x, *original_x, +original_x
        # Shapes
        N, C, D, H, W = x.shape
        DHW = D * H * W
        NC = N * C

        # Ensure contiguous memory layout
        x_contig = x.contiguous()
        out = torch.empty_like(x_contig)

        # Bias flattened to [C]
        bias_flat = self.bias.view(-1).contiguous()

        # Launch configuration
        # Keep the launch grid under the Ascend runtime limit for the target shape.
        BLOCK_K = 4096
        grid = (NC, triton.cdiv(DHW, BLOCK_K))

        _fused_bias_residual_mul_add_3d[grid](
            x_contig,
            bias_flat,
            out,
            DHW,
            C,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        return out


batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, stride, padding,
        output_padding, bias_shape
    ]
