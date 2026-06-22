import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def _mish_mish_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    zero = tl.zeros_like(x32)
    one = zero + 1.0
    twenty = zero + 20.0
    neg_twenty = zero - 20.0

    # Softplus with PyTorch's default threshold=20 for numerical parity:
    # softplus(x) = x if x > 20, ~exp(x) if x < -20, else max(x,0)+log1p(exp(-|x|))
    abs_x = tl.abs(x32)
    sp1_mid = tl.where(x32 > zero, x32, zero) + tl.log(one + tl.exp(-abs_x))
    sp1 = tl.where(x32 > twenty, x32,
                   tl.where(x32 < neg_twenty, tl.exp(x32), sp1_mid))

    tanh_sp1 = tl.tanh(sp1)
    mish1 = x32 * tanh_sp1

    # Second Mish
    abs_m1 = tl.abs(mish1)
    sp2_mid = tl.where(mish1 > zero, mish1,
                       zero) + tl.log(one + tl.exp(-abs_m1))
    sp2 = tl.where(mish1 > twenty, mish1,
                   tl.where(mish1 < neg_twenty, tl.exp(mish1), sp2_mid))

    tanh_sp2 = tl.tanh(sp2)
    out32 = mish1 * tanh_sp2

    out = out32.to(x.dtype)
    tl.store(y_ptr + offs, out, mask=mask)


def mish_mish_triton(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise ValueError("mish_mish_triton expects an Ascend NPU tensor")
    if x.requires_grad:
        raise ValueError(
            "mish_mish_triton does not support autograd-tracked tensors")
    if x.numel() == 0:
        return torch.empty_like(x)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "mish_mish_triton supports only float16, bfloat16, and float32 inputs"
        )

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    _mish_mish_kernel[grid](x_contig, y, n_elements)
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    The two Mish activations are fused into a single Triton kernel for speed.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = mish_mish_triton(x)
        return x


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
