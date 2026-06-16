import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _gelu_fwd_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    x3 = x32 * x32 * x32
    inner = 0.7978845608028654 * (x32 + 0.044715 * x3)
    abs_inner = tl.abs(inner)
    exp_term = tl.exp(-2.0 * abs_inner)
    tanh_abs = (1.0 - exp_term) / (1.0 + exp_term)
    tanh_inner = tl.where(inner >= 0.0, tanh_abs, -tanh_abs)
    y = (0.5 * x32 * (1.0 + tanh_inner)).to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a GELU activation using a Triton kernel on Ascend NPU tensors.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        supported_dtypes = {torch.float16, torch.float32, torch.bfloat16}
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-tracked inputs")

        x_contig = x.contiguous()
        n_elements = x_contig.numel()
        if n_elements == 0:
            return x_contig

        y = torch.empty_like(x_contig)
        block_size = 4096

        def grid(meta):
            return (triton.cdiv(n_elements, block_size), )

        _gelu_fwd_kernel[grid](
            x_contig.view(-1),
            y.view(-1),
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=8,
            num_stages=2,
        )

        return y


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device='npu')
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
