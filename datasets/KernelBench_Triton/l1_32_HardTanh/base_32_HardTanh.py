import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _hardtanh_kernel(x_ptr, y_ptr, n_elements, min_val, max_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Preserve NaN behavior exactly like PyTorch: if x is NaN, keep it as NaN.
    x = tl.where(x > max_val, max_val, tl.where(x < min_val, min_val, x))
    tl.store(y_ptr + offsets, x, mask=mask)


def _hardtanh_triton(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return y

    # Ascend's default launch cap is lower than the target-shape program count.
    os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _hardtanh_kernel[grid](
        x, y, n_elements, min_val, max_val,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,
        num_stages=1,
    )
    return y


class ModelNew(nn.Module):
    """
    HardTanh activation backed by the Triton kernel on Ascend NPU.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(x, "is_npu") or not x.is_npu:
            raise RuntimeError("ModelNew expects an input tensor on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")
        return _hardtanh_triton(x, -1.0, 1.0)
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
def get_init_inputs():
    return []  # No special initialization inputs needed
