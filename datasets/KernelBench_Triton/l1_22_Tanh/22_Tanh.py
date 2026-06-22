import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    abs_x = tl.abs(x32)
    exp_term = tl.exp(-2.0 * abs_x)
    tanh_abs = (1.0 - exp_term) / (1.0 + exp_term)
    y32 = tl.where(x32 >= 0.0, tanh_abs, -tanh_abs)
    y = y32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a Tanh activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
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
        BLOCK_SIZE = 4096

        def grid(meta):
            return (triton.cdiv(n_elements, BLOCK_SIZE), )

        _tanh_kernel[grid](
            x_contig,
            y,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
        return y.view_as(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
