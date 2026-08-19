import torch
import torch.nn as nn
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK_SIZE = 8192


@triton.jit
def _mish_direct_kernel(x_ptr, y_ptr, n_elements, THRESHOLD: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = x_in.to(tl.float32)
    use_large = x > THRESHOLD
    x_small = tl.where(use_large, 0.0, x)
    t = tl.exp(x_small)
    den = t * t + 2.0 * t + 2.0
    tanh_sp = tl.where(use_large, 1.0, 1.0 - 2.0 / den)
    y = (x * tanh_sp).to(x_in.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _mish_persistent_kernel(x_ptr, y_ptr, n_elements, n_programs,
                            THRESHOLD: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x_in.to(tl.float32)
        use_large = x > THRESHOLD
        x_small = tl.where(use_large, 0.0, x)
        t = tl.exp(x_small)
        den = t * t + 2.0 * t + 2.0
        tanh_sp = tl.where(use_large, 1.0, 1.0 - 2.0 / den)
        y = (x * tanh_sp).to(x_in.dtype)
        tl.store(y_ptr + offs, y, mask=mask)


def fused_softplus_tanh_mul(x: torch.Tensor) -> torch.Tensor:
    if not x.is_npu:
        raise RuntimeError(
            "fused_softplus_tanh_mul expects an Ascend NPU tensor")
    xi = x.contiguous()
    n = xi.numel()
    if n == 0:
        return torch.empty_like(xi)
    y = torch.empty_like(xi)
    n_tiles = triton.cdiv(n, _BLOCK_SIZE)
    if n_tiles > _MAX_PROGRAMS:
        n_programs = _MAX_PROGRAMS
        _mish_persistent_kernel[(n_programs, )](
            xi,
            y,
            n,
            n_programs,
            THRESHOLD=20.0,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    else:
        _mish_direct_kernel[(n_tiles, )](
            xi,
            y,
            n,
            THRESHOLD=20.0,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    return y


class ModelNew(nn.Module):
    """
    Conv2d -> Mish (x * tanh(softplus(x))) -> BatchNorm2d.
    The Mish activation uses Triton direct dispatch for normal grids and a
    persistent grid when the activation tile count exceeds Ascend FFTS limits.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 eps=1e-5,
                 momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = fused_softplus_tanh_mul(x)
        x = self.bn(x)
        return x


batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
