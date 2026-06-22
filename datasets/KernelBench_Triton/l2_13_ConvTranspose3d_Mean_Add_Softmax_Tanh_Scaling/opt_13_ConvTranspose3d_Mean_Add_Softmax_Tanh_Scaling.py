import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK_SIZE = 8192


@triton.jit
def _fill_const_direct(out_ptr, value, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    tl.store(out_ptr + offsets, value, mask=mask)


@triton.jit
def _fill_const_persistent(out_ptr, value, n_elements, n_programs,
                           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offsets = (tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(
            tl.int64)
        mask = offsets < n_elements
        tl.store(out_ptr + offsets, value, mask=mask)


def _to3(x):
    return (x, x, x) if isinstance(x, int) else tuple(x)


class ModelNew(nn.Module):
    """ConvTranspose3d -> mean(channel, keepdim=True) -> add -> softmax(dim=1) -> tanh -> scale.

    The channel mean produces C=1. Softmax over that singleton channel is exactly one
    for every output element, so the full chain is the constant tanh(1)*scaling_factor
    with only the ConvTranspose3d output spatial shape retained.
    """

    def __init__(
            self,
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_shape=(1, 1, 1, 1, 1),
            scaling_factor=2.0,
    ):
        super().__init__()
        # Baseline get_init_inputs passes scaling_factor as the sixth positional arg.
        if isinstance(bias_shape, (int, float)):
            scaling_factor = float(bias_shape)
            bias_shape = (1, 1, 1, 1, 1)

        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)
        self._k = _to3(self.conv_transpose.kernel_size)
        self._s = _to3(self.conv_transpose.stride)
        self._p = _to3(self.conv_transpose.padding)
        self._d = _to3(self.conv_transpose.dilation)
        self._op = _to3(self.conv_transpose.output_padding)
        self._const_val = math.tanh(1.0) * self.scaling_factor

    def forward(self, x):
        n, _, di, hi, wi = x.shape
        k, s, p, d, op = self._k, self._s, self._p, self._d, self._op
        do = (di - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + op[0] + 1
        ho = (hi - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + op[1] + 1
        wo = (wi - 1) * s[2] - 2 * p[2] + d[2] * (k[2] - 1) + op[2] + 1
        out = torch.empty((n, 1, do, ho, wo), device=x.device, dtype=x.dtype)
        n_elements = out.numel()
        n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
        if n_tiles > _MAX_PROGRAMS:
            _fill_const_persistent[(_MAX_PROGRAMS, )](out,
                                                      self._const_val,
                                                      n_elements,
                                                      _MAX_PROGRAMS,
                                                      BLOCK_SIZE=_BLOCK_SIZE,
                                                      num_stages=2)
        else:
            _fill_const_direct[(n_tiles, )](out,
                                            self._const_val,
                                            n_elements,
                                            BLOCK_SIZE=_BLOCK_SIZE,
                                            num_stages=2)
        return out


batch_size = 16
in_channels = 16
out_channels = 64
depth = 32
height = width = 128
kernel_size = 3
stride = 1
padding = 1
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, stride, padding, scaling_factor
    ]
