import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.math import tanh as tl_tanh

_MAX_GRID = 65535
_BLOCK_SIZE = 4096


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _tanh_mish_stable(x_f32):
    # tanh(softplus(x)) computed with one exp and no exp(2*x) overflow.
    z = tl.exp(-tl.abs(x_f32))
    z2 = z * z
    pos = (1.0 + 2.0 * z) / (1.0 + 2.0 * z + 2.0 * z2)
    neg = (z2 + 2.0 * z) / (z2 + 2.0 * z + 2.0)
    return tl.where(x_f32 >= 0.0, pos, neg)


@triton.jit
def _mish_tanh_direct_kernel(x_ptr, y_ptr, n_elements,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
    x_f32 = x.to(tl.float32)
    mish = x_f32 * _tanh_mish_stable(x_f32)
    out = tl_tanh(mish).to(x.dtype)
    tl.store(y_ptr + offs, out, mask=mask)


@triton.jit
def _mish_tanh_persistent_kernel(x_ptr, y_ptr, n_elements, n_programs,
                                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, n_tiles, n_programs, num_stages=2):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.multiple_of(offs, 16)
        tl.max_contiguous(offs, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
        x_f32 = x.to(tl.float32)
        mish = x_f32 * _tanh_mish_stable(x_f32)
        out = tl_tanh(mish).to(x.dtype)
        tl.store(y_ptr + offs, out, mask=mask)


def fused_mish_tanh(x: torch.Tensor) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError("fused_mish_tanh expects an Ascend NPU tensor")
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    if n_elements == 0:
        return y
    n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
    if n_tiles > _MAX_GRID:
        n_programs = _MAX_GRID
        _mish_tanh_persistent_kernel[(n_programs, )](x_contig,
                                                     y,
                                                     n_elements,
                                                     n_programs,
                                                     BLOCK_SIZE=_BLOCK_SIZE,
                                                     num_warps=8,
                                                     num_stages=2)
    else:
        _mish_tanh_direct_kernel[(n_tiles, )](x_contig,
                                              y,
                                              n_elements,
                                              BLOCK_SIZE=_BLOCK_SIZE,
                                              num_warps=8,
                                              num_stages=2)
    return y


class ModelNew(nn.Module):
    """3D convolution followed by fused Mish and Tanh activation."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew only supports Ascend NPU execution")
        return fused_mish_tanh(self.conv(x))


batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
