import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _clamp_divide_inplace_kernel(
    x_ptr,  # input/output pointer (contiguous tensor)
    n_elements,  # total number of elements
    min_value,  # clamp minimum (scalar)
    divisor,  # division scalar
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.multiple_of(offsets, 8)
    x = tl.load(x_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last")
    # Clamp to min then divide. Use maximum to match torch.clamp(min=...) semantics incl. NaN propagation.
    x = tl.maximum(x, min_value)
    x = x / divisor
    tl.store(x_ptr + offsets, x, mask=mask, eviction_policy="evict_last")


def _launch_clamp_divide_inplace(x: torch.Tensor, min_value: float, divisor: float):
    n_elements = x.numel()
    if n_elements == 0:
        return
    if divisor == 0:
        raise ValueError("divisor must be non-zero")
    if x.device.type != "npu":
        raise RuntimeError("conv_transpose3d_clamp_min_divide expects an Ascend NPU tensor")
    if not x.is_contiguous():
        x = x.contiguous()
    if n_elements >= (1 << 20):
        BLOCK_SIZE, WARPS, STAGES = 8192, 8, 2
    elif n_elements >= (1 << 18):
        BLOCK_SIZE, WARPS, STAGES = 4096, 4, 2
    else:
        BLOCK_SIZE, WARPS, STAGES = 2048, 4, 2
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _clamp_divide_inplace_kernel[grid](
        x, n_elements, float(min_value), float(divisor),
        BLOCK_SIZE=BLOCK_SIZE, num_warps=WARPS, num_stages=STAGES
    )


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = float(min_value)
        self.divisor = float(divisor)

    def forward(self, x):
        return conv_transpose3d_clamp_min_divide(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            min_value=self.min_value,
            divisor=self.divisor,
        )


def conv_transpose3d_clamp_min_divide(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    min_value: float = -1.0,
    divisor: float = 2.0,
) -> torch.Tensor:
    y = F.conv_transpose3d(x, weight, bias=bias, stride=stride, padding=padding)
    _launch_clamp_divide_inplace(y, min_value=min_value, divisor=divisor)
    return y
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]