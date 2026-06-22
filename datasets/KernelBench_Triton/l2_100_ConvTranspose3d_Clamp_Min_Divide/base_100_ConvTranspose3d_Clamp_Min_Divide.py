import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _clamp_divide_inplace_kernel(
    x_ptr,
    bias_ptr,
    n_elements,
    min_value,
    reciprocal,
    C: tl.constexpr,
    SPATIAL_VOL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    for block_idx in tl.static_range(NUM_BLOCKS_PER_PROG):
        offs_pid = pid * NUM_BLOCKS_PER_PROG + block_idx
        offsets = offs_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.multiple_of(offsets, 8)
        x = tl.load(x_ptr + offsets,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_last")
        chan_idx = (offsets // SPATIAL_VOL) % C
        bias_val = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0)
        x = x + bias_val
        x = tl.maximum(x, min_value, propagate_nan=tl.PropagateNan.ALL)
        x = x * reciprocal
        tl.store(x_ptr + offsets, x, mask=mask, eviction_policy="evict_last")


def _launch_clamp_divide_inplace(x: torch.Tensor, bias: torch.Tensor,
                                 min_value: float, divisor: float):
    n_elements = x.numel()
    if n_elements == 0:
        return
    if divisor == 0:
        raise ValueError("divisor must be non-zero")
    if x.device.type != "npu":
        raise RuntimeError(
            "conv_transpose3d_clamp_min_divide expects an Ascend NPU tensor")
    if not x.is_contiguous():
        x = x.contiguous()
    reciprocal = 1.0 / divisor
    if n_elements >= (1 << 20):
        BLOCK_SIZE, WARPS, STAGES, NUM_BLOCKS = 8192, 8, 3, 2
    elif n_elements >= (1 << 18):
        BLOCK_SIZE, WARPS, STAGES, NUM_BLOCKS = 4096, 4, 3, 2
    else:
        BLOCK_SIZE, WARPS, STAGES, NUM_BLOCKS = 2048, 4, 3, 2
    C = x.shape[1]
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    SPATIAL_VOL = D * H * W
    grid = (triton.cdiv(n_elements, BLOCK_SIZE * NUM_BLOCKS), )
    _clamp_divide_inplace_kernel[grid](x,
                                       bias,
                                       n_elements,
                                       float(min_value),
                                       float(reciprocal),
                                       C=C,
                                       SPATIAL_VOL=SPATIAL_VOL,
                                       BLOCK_SIZE=BLOCK_SIZE,
                                       NUM_BLOCKS_PER_PROG=NUM_BLOCKS,
                                       num_warps=WARPS,
                                       num_stages=STAGES)


class ModelNew(nn.Module):

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
    y = F.conv_transpose3d(x,
                           weight,
                           bias=None,
                           stride=stride,
                           padding=padding)
    if bias is not None:
        _launch_clamp_divide_inplace(y,
                                     bias,
                                     min_value=min_value,
                                     divisor=divisor)
    else:
        n_elements = y.numel()
        if n_elements == 0:
            return y
        if y.device.type != "npu":
            raise RuntimeError(
                "conv_transpose3d_clamp_min_divide expects an Ascend NPU tensor"
            )
        if not y.is_contiguous():
            y = y.contiguous()
        reciprocal = 1.0 / divisor
        if n_elements >= (1 << 20):
            BLOCK_SIZE, WARPS, STAGES, NUM_BLOCKS = 8192, 8, 3, 2
        elif n_elements >= (1 << 18):
            BLOCK_SIZE, WARPS, STAGES, NUM_BLOCKS = 4096, 4, 3, 2
        else:
            BLOCK_SIZE, WARPS, STAGES, NUM_BLOCKS = 2048, 4, 3, 2
        grid = (triton.cdiv(n_elements, BLOCK_SIZE * NUM_BLOCKS), )
        _clamp_divide_no_bias_kernel[grid](y,
                                           n_elements,
                                           float(min_value),
                                           float(reciprocal),
                                           BLOCK_SIZE=BLOCK_SIZE,
                                           NUM_BLOCKS_PER_PROG=NUM_BLOCKS,
                                           num_warps=WARPS,
                                           num_stages=STAGES)
    return y


@triton.jit
def _clamp_divide_no_bias_kernel(
    x_ptr,
    n_elements,
    min_value,
    reciprocal,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    for block_idx in tl.static_range(NUM_BLOCKS_PER_PROG):
        offs_pid = pid * NUM_BLOCKS_PER_PROG + block_idx
        offsets = offs_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.multiple_of(offsets, 8)
        x = tl.load(x_ptr + offsets,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_last")
        x = tl.maximum(x, min_value, propagate_nan=tl.PropagateNan.ALL)
        x = x * reciprocal
        tl.store(x_ptr + offsets, x, mask=mask, eviction_policy="evict_last")


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
