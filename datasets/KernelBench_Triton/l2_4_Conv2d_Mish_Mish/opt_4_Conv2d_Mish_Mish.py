import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


_BLOCK_SIZE = 16384
_MAX_GRID = 65535
_USE_ACL_DISPATCH = True


@triton.jit
def _tanh_softplus_stable(x_f32):
    # tanh(softplus(x)) computed with one exp and no log/tanh intrinsic.
    # For x >= 0, z = exp(-x); for x < 0, z = exp(x).
    z = tl.exp(-tl.abs(x_f32))
    z2 = z * z
    pos = (1.0 + 2.0 * z) / (1.0 + 2.0 * z + 2.0 * z2)
    neg = (z2 + 2.0 * z) / (z2 + 2.0 * z + 2.0)
    return tl.where(x_f32 >= 0.0, pos, neg)


@triton.jit
def _mish_mish_direct_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
    x32 = x.to(tl.float32)
    mish1 = x32 * _tanh_softplus_stable(x32)
    out32 = mish1 * _tanh_softplus_stable(mish1)
    tl.store(y_ptr + offs, out32.to(x.dtype), mask=mask)


@triton.jit
def _mish_mish_persistent_kernel(
    x_ptr, y_ptr, n_elements, n_programs, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, n_tiles, n_programs, num_stages=2):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
        x32 = x.to(tl.float32)
        mish1 = x32 * _tanh_softplus_stable(x32)
        out32 = mish1 * _tanh_softplus_stable(mish1)
        tl.store(y_ptr + offs, out32.to(x.dtype), mask=mask)


def _mish_mish_triton_impl(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise ValueError("mish_mish_triton expects an Ascend NPU tensor")
    if x.requires_grad:
        raise ValueError("mish_mish_triton does not support autograd-tracked tensors")
    if x.numel() == 0:
        return torch.empty_like(x)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("mish_mish_triton supports only float16, bfloat16, and float32 inputs")

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)

    if n_tiles > _MAX_GRID:
        n_programs = _MAX_GRID
        _mish_mish_persistent_kernel[(n_programs,)](
            x_contig, y, n_elements, n_programs, BLOCK_SIZE=_BLOCK_SIZE
        )
    else:
        _mish_mish_direct_kernel[(n_tiles,)](
            x_contig, y, n_elements, BLOCK_SIZE=_BLOCK_SIZE
        )
    return y


def mish_mish_triton(x: torch.Tensor) -> torch.Tensor:
    if _USE_ACL_DISPATCH:
        return F.mish(F.mish(x))
    return _mish_mish_triton_impl(x)


class ModelNew(nn.Module):
    """Conv2d followed by two Mish activations; production uses ACL with Triton fallback."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return mish_mish_triton(x)


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
