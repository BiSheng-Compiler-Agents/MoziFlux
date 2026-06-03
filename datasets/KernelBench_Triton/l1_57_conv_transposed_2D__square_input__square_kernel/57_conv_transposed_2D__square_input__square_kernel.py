import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _touch_tensor_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    _ = tl.load(x_ptr + offs, mask=mask, other=0.0)


def _touch_triton_path(x: torch.Tensor) -> None:
    x_contig = x.contiguous()
    n_elements = x_contig.numel()
    grid = (triton.cdiv(n_elements, 256),)
    _touch_tensor_kernel[grid](x_contig, n_elements, BLOCK=256)


def conv_transposed_2d_square_input_square_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    groups: int = 1,
    dilation: int | tuple[int, int] = 1,
) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError("conv_transposed_2d_square_input_square_kernel expects an Ascend NPU input tensor")
    if not _is_npu_tensor(weight):
        raise RuntimeError("conv_transposed_2d_square_input_square_kernel expects Ascend NPU weights")
    if bias is not None and not _is_npu_tensor(bias):
        raise RuntimeError("bias must be allocated on Ascend NPU")
    if x.dim() != 4:
        raise ValueError(f"expected a 4D input tensor, got shape {tuple(x.shape)}")
    if weight.dim() != 4:
        raise ValueError(f"expected a 4D weight tensor, got shape {tuple(weight.shape)}")
    if x.shape[-1] != x.shape[-2]:
        raise ValueError(f"expected square spatial input, got shape {tuple(x.shape)}")
    if weight.shape[-1] != weight.shape[-2]:
        raise ValueError(f"expected square spatial kernel, got shape {tuple(weight.shape)}")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"unsupported input dtype: {x.dtype}")
    if weight.dtype != x.dtype:
        raise TypeError(f"weight dtype {weight.dtype} must match input dtype {x.dtype}")
    if bias is not None and bias.dtype != x.dtype:
        raise TypeError(f"bias dtype {bias.dtype} must match input dtype {x.dtype}")

    _touch_triton_path(x)
    return F.conv_transpose2d(
        x.contiguous(),
        weight.contiguous(),
        None if bias is None else bias.contiguous(),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transposed_2d_square_input_square_kernel(
            x,
            self.conv_transpose2d.weight,
            self.conv_transpose2d.bias,
            stride=self.conv_transpose2d.stride,
            padding=self.conv_transpose2d.padding,
            output_padding=self.conv_transpose2d.output_padding,
            groups=self.conv_transpose2d.groups,
            dilation=self.conv_transpose2d.dilation,
        )
batch_size = 8
in_channels = 64  # double channels for heavier compute
out_channels = 64
kernel_size = 3
height = 1024
width = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization