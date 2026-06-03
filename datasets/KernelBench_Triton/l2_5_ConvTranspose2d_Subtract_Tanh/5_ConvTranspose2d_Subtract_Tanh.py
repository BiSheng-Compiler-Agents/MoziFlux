import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _bias_sub_tanh_kernel(
    x_ptr,         # *T: input/output tensor (N, C, H, W) flattened
    b_ptr,         # *T: bias tensor (C)
    y_ptr,         # *T: output tensor (same as x)
    HW: tl.constexpr,   # H * W
    C: tl.constexpr,    # number of channels
    NCHW: tl.constexpr, # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NCHW

    # Compute channel index for each element in flattened NCHW layout
    c_idx = (offs // HW) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + c_idx, mask=mask, other=0.0)

    x32 = x.to(tl.float32)
    b32 = b.to(tl.float32)
    z = x32 - b32
    y = tl.tanh(z)

    # Store back; Triton will cast to the destination pointer dtype if needed
    tl.store(y_ptr + offs, y, mask=mask)


def _bias_sub_tanh_fused(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("The fused Triton kernel only supports Ascend NPU tensors.")

    x = x.contiguous()
    b = bias.reshape(-1).to(device=x.device, dtype=x.dtype)
    y = torch.empty_like(x)

    N, C, H, W = x.shape
    n_elements = x.numel()
    HW = H * W

    # Launch configuration
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _bias_sub_tanh_kernel[grid](
        x, b, y,
        HW, C, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return y


def conv_transpose2d_subtract_tanh(
    x: torch.Tensor,
    weight: torch.Tensor,
    subtract_bias: torch.Tensor,
    conv_bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 2,
    padding: int | tuple[int, int] = 1,
    output_padding: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("conv_transpose2d_subtract_tanh expects input on Ascend NPU.")
    if weight.device.type != "npu":
        raise RuntimeError("conv_transpose2d_subtract_tanh expects weight on Ascend NPU.")
    if subtract_bias.device.type != "npu":
        raise RuntimeError("conv_transpose2d_subtract_tanh expects subtract_bias on Ascend NPU.")
    if conv_bias is not None and conv_bias.device.type != "npu":
        raise RuntimeError("conv_transpose2d_subtract_tanh expects conv_bias on Ascend NPU.")

    x = F.conv_transpose2d(
        x,
        weight,
        bias=conv_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    return _bias_sub_tanh_fused(x, subtract_bias)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return conv_transpose2d_subtract_tanh(
            x,
            self.conv_transpose.weight,
            self.bias,
            conv_bias=self.conv_transpose.bias,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            dilation=self.conv_transpose.dilation,
            groups=self.conv_transpose.groups,
        )
batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]