import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gelu_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    c = 0.7978845608028654
    ca = 0.035677408136300125
    x2 = x_f32 * x_f32
    u = x_f32 * (c + ca * x2)
    s = 1.0 / (1.0 + tl.math.exp(-2.0 * u))
    y_f32 = tl.math.fma(x_f32, s, 0.0)
    y = y_f32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


def _gelu_tanh_triton(x):
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    if n_elements == 0:
        return y.view_as(x)
    BLOCK_SIZE = 18432

    def grid(META):
        return (triton.cdiv(n_elements, META['BLOCK_SIZE']), )

    _gelu_tanh_kernel[grid](x_contig, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y.view_as(x)


class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU.")
        return _gelu_tanh_triton(x)


batch_size = 8192
dim = 8192


def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
