import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mish_mish_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    clamp_x = tl.where(x32 < 80.0, x32, 80.0)
    ex = tl.exp(clamp_x) + 1.0
    ex2 = ex * ex
    mish1 = x32 * (ex2 - 1.0) / (ex2 + 1.0)
    clamp_m1 = tl.where(mish1 < 80.0, mish1, 80.0)
    ex_b = tl.exp(clamp_m1) + 1.0
    ex2_b = ex_b * ex_b
    out32 = mish1 * (ex2_b - 1.0) / (ex2_b + 1.0)
    out = out32.to(x.dtype)
    tl.store(y_ptr + offs, out, mask=mask)


def mish_mish_triton(x: torch.Tensor) -> torch.Tensor:
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
    if n_elements > 1048576:
        BLOCK_SIZE = 10240
    else:
        BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _mish_mish_kernel[grid](x_contig, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y


class ModelNew(nn.Module):
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
