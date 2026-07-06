import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


_MAX_GRID = 65535
_BLOCK_SIZE = 8192


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _div_leakyrelu_direct_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    inv_div,
    neg_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_SIZE)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
    inv = tl.full((), inv_div, x.dtype)
    slope = tl.full((), neg_slope, x.dtype)
    zero = tl.full((), 0.0, x.dtype)

    y = x * inv
    out = tl.where(y >= zero, y, y * slope)
    tl.store(y_ptr + offs, out, mask=mask)


@triton.jit
def _div_leakyrelu_persistent_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    n_programs,
    inv_div,
    neg_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)

    for tile_id in range(pid, n_tiles, n_programs):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        tl.multiple_of(offs, 16)
        tl.max_contiguous(offs, BLOCK_SIZE)

        x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
        inv = tl.full((), inv_div, x.dtype)
        slope = tl.full((), neg_slope, x.dtype)
        zero = tl.full((), 0.0, x.dtype)

        y = x * inv
        out = tl.where(y >= zero, y, y * slope)
        tl.store(y_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    """
    Conv2d followed by division and LeakyReLU. The Conv2d remains on ACL/PyTorch-NPU;
    the pointwise epilogue is fused into one Triton vector kernel.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        divisor=2,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects Ascend NPU tensors")

        conv_out = self.conv(x).contiguous()
        out = torch.empty_like(conv_out)
        n_elements = out.numel()
        if n_elements == 0:
            return out

        inv_div = float(1.0 / float(self.divisor))
        neg_slope = 0.01
        n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)

        if n_tiles <= _MAX_GRID:
            _div_leakyrelu_direct_kernel[(n_tiles,)](
                conv_out, out, n_elements, inv_div, neg_slope, BLOCK_SIZE=_BLOCK_SIZE
            )
        else:
            n_programs = _MAX_GRID
            _div_leakyrelu_persistent_kernel[(n_programs,)](
                conv_out,
                out,
                n_elements,
                n_programs,
                inv_div,
                neg_slope,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
        return out


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]
