import math
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _softmax_row_fwd_db_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x_row_ptr = x_ptr + pid * stride_row
    y_row_ptr = y_ptr + pid * stride_row

    x = tl.load(x_row_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(x, axis=0)
    numerators = tl.exp(x - row_max)
    denom = tl.sum(numerators, axis=0)
    y = numerators / denom
    tl.store(y_row_ptr + offs, y, mask=mask)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _select_kernel_config(n_cols: int):
    block_size = _next_power_of_2(n_cols)
    if n_cols >= 16384:
        return block_size, 16, 2
    if n_cols >= 8192:
        return block_size, 8, 2
    if n_cols >= 4096:
        return block_size, 8, 2
    return block_size, 4, 2


class ModelNew(nn.Module):
    """
    Simple model that performs a row-wise Softmax activation using a Triton Ascend kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("ModelNew expects a 2D tensor shaped [batch, columns]")
        if not getattr(x, "is_npu", False):
            raise RuntimeError("ModelNew requires input tensors on Ascend NPU")

        # Ensure contiguous for coalesced access
        x_in = x.contiguous()
        B, D = x_in.shape
        y_out = torch.empty_like(x_in)

        BLOCK_SIZE, num_warps, num_stages = _select_kernel_config(D)
        grid = (B,)

        _softmax_row_fwd_db_kernel[grid](
            x_in, y_out,
            D,
            x_in.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y_out
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
def get_init_inputs():
    return []  # No special initialization inputs needed