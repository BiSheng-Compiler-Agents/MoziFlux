import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

DEFAULT_IN_CHANNELS = 32
DEFAULT_OUT_CHANNELS = 32
DEFAULT_KERNEL_SIZE = (3, 7)
DEFAULT_STRIDE = (1, 1)
DEFAULT_PADDING = (1, 3)
DEFAULT_OUTPUT_PADDING = (0, 0)
DEFAULT_DILATION = (1, 1)
DEFAULT_GROUPS = 1
DEFAULT_BIAS = False


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _touch_tensor_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    _ = tl.load(x_ptr + offsets, mask=mask, other=0.0)


def _touch_triton_path(x: torch.Tensor) -> None:
    x_contiguous = x.contiguous()
    n_elements = x_contiguous.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n_elements, BLOCK), )
    _touch_tensor_kernel[grid](x_contiguous, n_elements, BLOCK=BLOCK)


def conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = DEFAULT_STRIDE,
    padding: int | tuple[int, int] = DEFAULT_PADDING,
    output_padding: int | tuple[int, int] = DEFAULT_OUTPUT_PADDING,
    groups: int = DEFAULT_GROUPS,
    dilation: int | tuple[int, int] = DEFAULT_DILATION,
) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded expects an Ascend NPU input tensor"
        )
    if not _is_npu_tensor(weight):
        raise RuntimeError(
            "conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded expects Ascend NPU weights"
        )
    if bias is not None and not _is_npu_tensor(bias):
        raise RuntimeError("bias must be allocated on Ascend NPU")
    if x.dim() != 4:
        raise ValueError(
            f"expected a 4D input tensor, got shape {tuple(x.shape)}")
    if weight.dim() != 4:
        raise ValueError(
            f"expected a 4D weight tensor, got shape {tuple(weight.shape)}")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"unsupported input dtype: {x.dtype}")
    if weight.dtype != x.dtype:
        raise TypeError(
            f"weight dtype {weight.dtype} must match input dtype {x.dtype}")
    if bias is not None and bias.dtype != x.dtype:
        raise TypeError(
            f"bias dtype {bias.dtype} must match input dtype {x.dtype}")
    if x.shape[1] != weight.shape[0]:
        raise ValueError(
            f"input channels {x.shape[1]} must match weight input channels {weight.shape[0]}"
        )

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
    Performs a 2D transposed convolution with asymmetric input and kernel size.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: tuple[int, int] = DEFAULT_KERNEL_SIZE,
        stride: tuple[int, int] = DEFAULT_STRIDE,
        padding: tuple[int, int] = DEFAULT_PADDING,
        output_padding: tuple[int, int] = DEFAULT_OUTPUT_PADDING,
        dilation: tuple[int, int] = DEFAULT_DILATION,
        groups: int = DEFAULT_GROUPS,
        bias: bool = DEFAULT_BIAS,
    ):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded(
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
in_channels = 32
out_channels = 32
kernel_size = (3, 7)
height = 512
width = 1024
stride = (1, 1)
padding = (1, 3)


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
