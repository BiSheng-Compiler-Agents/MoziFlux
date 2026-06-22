import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK_SIZE = 8192


@triton.jit
def _clamp_divide_direct_kernel(
    x_ptr,
    n_elements,
    min_value,
    inv_divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last")
    x = tl.maximum(x, min_value) * inv_divisor
    tl.store(x_ptr + offsets, x, mask=mask, eviction_policy="evict_last")


@triton.jit
def _clamp_divide_persistent_kernel(
    x_ptr,
    n_elements,
    n_programs,
    min_value,
    inv_divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offsets = (tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(
            tl.int64)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_last")
        x = tl.maximum(x, min_value) * inv_divisor
        tl.store(x_ptr + offsets, x, mask=mask, eviction_policy="evict_last")


def _launch_clamp_divide_inplace(x: torch.Tensor, min_value: float,
                                 divisor: float):
    n_elements = x.numel()
    if n_elements == 0:
        return x
    if divisor == 0:
        raise ValueError("divisor must be non-zero")
    if x.device.type != "npu":
        raise RuntimeError(
            "conv_transpose3d_clamp_min_divide expects an Ascend NPU tensor")
    if not x.is_contiguous():
        x = x.contiguous()

    inv_divisor = 1.0 / float(divisor)
    n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
    if n_tiles > _MAX_PROGRAMS:
        n_programs = _MAX_PROGRAMS
        _clamp_divide_persistent_kernel[(n_programs, )](
            x,
            n_elements,
            n_programs,
            float(min_value),
            inv_divisor,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
    else:
        _clamp_divide_direct_kernel[(n_tiles, )](
            x,
            n_elements,
            float(min_value),
            inv_divisor,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
    return x


class ModelNew(nn.Module):
    """
    Transposed 3D convolution followed by clamp-min and divide.

    Optimization: preserve ACL ConvTranspose3d, then run a legal two-path Triton
    epilogue. The large target output exceeds Ascend's 65,535 launch-grid cap, so
    the epilogue uses a persistent tile loop only for that oversized path.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding)
        self.min_value = float(min_value)
        self.divisor = float(divisor)

    def forward(self, x):
        y = F.conv_transpose3d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
        )
        return _launch_clamp_divide_inplace(y, self.min_value, self.divisor)


def conv_transpose3d_clamp_min_divide(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    min_value: float = -1.0,
    divisor: float = 2.0,
) -> torch.Tensor:
    y = F.conv_transpose3d(x,
                           weight,
                           bias=bias,
                           stride=stride,
                           padding=padding)
    return _launch_clamp_divide_inplace(y,
                                        min_value=min_value,
                                        divisor=divisor)


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
    return [
        in_channels, out_channels, kernel_size, stride, padding, min_value,
        divisor
    ]
