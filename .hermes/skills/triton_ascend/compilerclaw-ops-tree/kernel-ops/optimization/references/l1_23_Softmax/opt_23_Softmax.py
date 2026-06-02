import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _softmax_row_fwd_db_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    stride_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    row_max = tl.where(row_mask, -float("inf"), 0.0)
    row_sum = tl.where(row_mask, 0.0, 1.0)
    n_loop = n_cols

    for start in range(0, n_loop, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        offs = tl.multiple_of(offs, BLOCK_N)
        offs = tl.max_contiguous(offs, BLOCK_N)
        col_mask = offs < n_cols
        mask = row_mask[:, None] & col_mask[None, :]
        x_ptrs = x_ptr + rows[:, None] * stride_row + offs[None, :]
        x = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=1)
        new_row_max = tl.where(row_mask, tl.maximum(row_max, block_max), 0.0)
        exp_scale = tl.where(row_mask, tl.exp(row_max - new_row_max), 0.0)
        row_sum = tl.where(
            row_mask,
            row_sum * exp_scale + tl.sum(tl.exp(x - new_row_max[:, None]), axis=1),
            1.0,
        )
        row_max = new_row_max

    for start in range(0, n_loop, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        offs = tl.multiple_of(offs, BLOCK_N)
        offs = tl.max_contiguous(offs, BLOCK_N)
        col_mask = offs < n_cols
        mask = row_mask[:, None] & col_mask[None, :]
        x_ptrs = x_ptr + rows[:, None] * stride_row + offs[None, :]
        y_ptrs = y_ptr + rows[:, None] * stride_row + offs[None, :]
        x = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        y = tl.exp(x - row_max[:, None]) / row_sum[:, None]
        tl.store(y_ptrs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("ModelNew expects a 2D tensor shaped [batch, columns]")
        if not getattr(x, "is_npu", False):
            raise RuntimeError("ModelNew requires input tensors on Ascend NPU")

        x_in = x.contiguous()
        n_rows, n_cols = x_in.shape
        y_out = torch.empty_like(x_in)
        grid = (triton.cdiv(n_rows, 2),)
        _softmax_row_fwd_db_kernel[grid](
            x_in,
            y_out,
            n_rows,
            n_cols,
            x_in.stride(0),
            BLOCK_M=2,
            BLOCK_N=2048,
            num_warps=8,
            num_stages=1,
        )
        return y_out


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
