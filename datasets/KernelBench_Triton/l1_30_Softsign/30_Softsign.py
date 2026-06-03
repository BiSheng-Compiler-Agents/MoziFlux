import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _softsign_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Compute in input dtype to avoid unnecessary upcasts
    one = tl.full([1], 1.0, dtype=x.dtype)
    y = x / (tl.abs(x) + one)
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a Softsign activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softsign activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softsign applied, same shape as input.
        """
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(f"Unsupported dtype for ModelNew: {x.dtype}")

        x_contiguous = x.contiguous()
        y = torch.empty_like(x_contiguous)
        n_elements = x_contiguous.numel()
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)

        _softsign_kernel[grid](
            x_contiguous.reshape(-1),
            y.reshape(-1),
            n_elements,
            BLOCK_SIZE=block_size,
        )
        return y.reshape_as(x)
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device='npu')
    return [x]
def get_init_inputs():
    return []  # No special initialization inputs needed