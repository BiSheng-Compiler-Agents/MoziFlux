import torch
import torch.nn as nn
import triton
import triton.language as tl

DEFAULT_IN_CHANNELS = 32
DEFAULT_OUT_CHANNELS = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 1
DEFAULT_OUTPUT_PADDING = 1
DEFAULT_BIAS_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1, 1, 1)

_MAX_PROGRAMS = 65535
_BLOCK_SIZE = 4096


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _fused_add_hswish_direct_kernel(x_ptr, add_ptr, out_ptr, n_elements,
                                    BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    add = tl.load(add_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = x + add
    clipped = tl.minimum(tl.maximum(z + 3.0, 0.0), 6.0)
    out = z * clipped * 0.16666666666666666
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _fused_add_hswish_persistent_kernel(x_ptr, add_ptr, out_ptr, n_elements,
                                        n_programs, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in range(pid, n_tiles, n_programs):
        offs = (tile_id * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        add = tl.load(add_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        z = x + add
        clipped = tl.minimum(tl.maximum(z + 3.0, 0.0), 6.0)
        out = z * clipped * 0.16666666666666666
        tl.store(out_ptr + offs, out, mask=mask)


def _fused_add_hswish_mul(x: torch.Tensor,
                          add_input: torch.Tensor) -> torch.Tensor:
    if not _is_npu_tensor(x) or not _is_npu_tensor(add_input):
        raise RuntimeError("_fused_add_hswish_mul expects Ascend NPU tensors.")
    if x.device != add_input.device:
        raise RuntimeError("x and add_input must be on the same device.")
    if x.shape != add_input.shape:
        raise RuntimeError("x and add_input must have the same shape.")
    if x.dtype != add_input.dtype:
        raise RuntimeError("x and add_input must have the same dtype.")

    x_c = x.contiguous()
    add_c = add_input.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    if n_elements == 0:
        return out

    n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
    if n_tiles > _MAX_PROGRAMS:
        _fused_add_hswish_persistent_kernel[(_MAX_PROGRAMS, )](
            x_c, add_c, out, n_elements, _MAX_PROGRAMS, BLOCK=_BLOCK_SIZE)
    else:
        _fused_add_hswish_direct_kernel[(n_tiles, )](x_c,
                                                     add_c,
                                                     out,
                                                     n_elements,
                                                     BLOCK=_BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs ConvTranspose3d, adds an input tensor, and applies HardSwish:
      out = hardswish(convT(x) + add_input)
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        output_padding: int = DEFAULT_OUTPUT_PADDING,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        if not _is_npu_tensor(x) or not _is_npu_tensor(add_input):
            raise RuntimeError("ModelNew expects Ascend NPU input tensors.")
        x = self.conv_transpose(x)
        return _fused_add_hswish_mul(x, add_input)


batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)


def get_inputs():
    return [
        torch.rand(batch_size, in_channels, D, H, W),
        torch.rand(batch_size, out_channels, D * stride, H * stride,
                   W * stride)
    ]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, stride, padding,
        output_padding, bias_shape
    ]
