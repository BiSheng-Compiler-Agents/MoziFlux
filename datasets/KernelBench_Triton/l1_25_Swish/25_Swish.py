import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _swish_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    sigmoid = tl.sigmoid(xf)
    y = (xf * sigmoid).to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a Swish activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-tracked inputs")

        x_contig = x.contiguous()
        n_elements = x_contig.numel()
        if n_elements == 0:
            return x_contig

        y = torch.empty_like(x_contig)
        block_size = 4096
        num_warps = 8

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _swish_kernel[grid](
            x_contig,
            y,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=2,
        )

        return y
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
def get_init_inputs():
    return []  # No special initialization inputs needed