import torch
import torch.nn as nn
import triton
import triton.language as tl


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")

@triton.jit
def _mish_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)
    mask = offs < n_elements

    # Load and upcast to fp32 for numerical stability
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # tanh(softplus(x)) via one exp(-|x|) (avoids overflow vs separate exp(-x), exp(x)):
    #   r = exp(-|x|)  in (0,1]
    #   x >= 0: r = exp(-x) => tanh(softplus) = (1+2r) / (1+2r+2r^2)
    #   x < 0:  r = exp(x)  => tanh(softplus) = r(r+2) / (r(r+2)+2)
    r = tl.exp(-tl.abs(x_f32))
    two_r = 2.0 * r
    pos_num = 1.0 + two_r
    pos_den = 1.0 + two_r + 2.0 * r * r
    neg_num = r * (r + 2.0)
    neg_den = neg_num + 2.0
    tanh_sp = tl.where(x_f32 >= 0.0, pos_num / pos_den, neg_num / neg_den)

    # mish(x) = x * tanh(softplus(x))
    mish = x_f32 * tanh_sp

    out_f32 = (tl.exp(2.0 * mish) - 1.0) / (tl.exp(2.0 * mish) + 1.0)

    # Downcast and store
    out = out_f32.to(x.dtype)
    tl.store(y_ptr + offs, out, mask=mask)


def fused_mish_tanh(x: torch.Tensor) -> torch.Tensor:
    # Fused activation: y = tanh(mish(x)) with stable softplus
    if not _is_npu_tensor(x):
        raise RuntimeError("fused_mish_tanh expects an Ascend NPU tensor")
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    if n_elements == 0:
        return y
    BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _mish_tanh_kernel[grid](x_contig, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
    return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew only supports Ascend NPU execution")
        x = self.conv(x)
        # Fused Triton kernel for Mish + Tanh
        x = fused_mish_tanh(x)
        return x
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
