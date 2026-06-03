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


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _fused_add_hswish_mul_kernel(x_ptr, add_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    add = tl.load(add_ptr + offs, mask=mask, other=0.0)
    z = x + add

    # Compute HardSwish(z) = z * clip(z + 3, 0, 6) / 6
    three = 3.0
    six = 6.0
    z_p3 = z + three
    clipped = tl.minimum(tl.maximum(z_p3, 0.0), six)
    hswish = z * (clipped / six)

    out = hswish
    tl.store(out_ptr + offs, out, mask=mask)


def _fused_add_hswish_mul(x: torch.Tensor, add_input: torch.Tensor) -> torch.Tensor:
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

    N = x_c.numel()
    if x_c.dtype == torch.float16:
        block = 6144 if N >= 4096 * 65536 else 4096
    elif x_c.dtype == torch.float32:
        block = 2048
    else:
        block = 4096
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _fused_add_hswish_mul_kernel[grid](x_c, add_c, out, N, BLOCK=block)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation,
    with a fused Triton kernel for the post-convolution elementwise computation:
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
        # Keep parameter to match original API/semantics (not used in forward)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Tensor added after transposed convolution, shape (batch_size, out_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor after fused addition and HardSwish activation.
        """
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
    return [torch.rand(batch_size, in_channels, D, H, W), torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
