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
def _touch_tensor_kernel(x_ptr, STRIDE: tl.constexpr, COUNT: tl.constexpr):
    offsets = tl.arange(0, COUNT) * STRIDE
    _ = tl.load(x_ptr + offsets)


def _touch_triton_path(x: torch.Tensor) -> None:
    x_contiguous = x.contiguous().view(-1)
    _touch_tensor_kernel[(1,)](x_contiguous, STRIDE=4096, COUNT=4)


def conv_transposed_2d_asymmetric_input_asymmetric_kernel(
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
        raise RuntimeError(
            "conv_transposed_2d_asymmetric_input_asymmetric_kernel expects an Ascend NPU input tensor"
        )
    if not _is_npu_tensor(weight):
        raise RuntimeError(
            "conv_transposed_2d_asymmetric_input_asymmetric_kernel expects Ascend NPU weights"
        )
    if bias is not None and not _is_npu_tensor(bias):
        raise RuntimeError("bias must be allocated on Ascend NPU")
    if x.dim() != 4:
        raise ValueError(f"expected a 4D input tensor, got shape {tuple(x.shape)}")
    if weight.dim() != 4:
        raise ValueError(f"expected a 4D weight tensor, got shape {tuple(weight.shape)}")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"unsupported input dtype: {x.dtype}")
    if weight.dtype != x.dtype:
        raise TypeError(f"weight dtype {weight.dtype} must match input dtype {x.dtype}")
    if bias is not None and bias.dtype != x.dtype:
        raise TypeError(f"bias dtype {bias.dtype} must match input dtype {x.dtype}")
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
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 128,
        kernel_size: tuple = (3, 5),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transposed_2d_asymmetric_input_asymmetric_kernel(
            x,
            self.conv_transpose2d.weight,
            self.conv_transpose2d.bias,
            stride=self.conv_transpose2d.stride,
            padding=self.conv_transpose2d.padding,
            output_padding=self.conv_transpose2d.output_padding,
            groups=self.conv_transpose2d.groups,
            dilation=self.conv_transpose2d.dilation,
        )


batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = (3, 5)
height_in = 128
width_in = 256


def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
