import torch
import torch.nn as nn
import torch_npu  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def _rcumsum_lastdim_kernel(
    x_ptr, y_ptr,
    rows, N,
    stride_x_row, stride_x_col,
    stride_y_row, stride_y_col,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per row
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    # Row base pointers
    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    # Lane indices within a tile
    i = tl.arange(0, BLOCK_N)

    # Accumulator that propagates across tiles from right to left
    carry = tl.zeros((), dtype=tl.float32)

    # Prefetch the rightmost tile
    last_block = NUM_BLOCKS - 1
    base = last_block * BLOCK_N
    rev_cols = base + (BLOCK_N - 1 - i)
    m_rev = rev_cols < N
    x_rev = tl.load(row_x_ptr + rev_cols * stride_x_col, mask=m_rev, other=0.0).to(tl.float32)

    # Process tiles from rightmost to leftmost with simple software pipelining
    for b in range(NUM_BLOCKS):
        # Compute for the prefetched tile
        block_idx = NUM_BLOCKS - 1 - b
        base_cur = block_idx * BLOCK_N
        rev_cols_cur = base_cur + (BLOCK_N - 1 - i)
        m_rev_cur = rev_cols_cur < N

        # Inclusive scan within the reversed tile -> reverse-cumsum on original
        scan_rev = tl.cumsum(x_rev, axis=0)
        out_rev = scan_rev + carry
        tl.store(row_y_ptr + rev_cols_cur * stride_y_col, out_rev.to(tl.float32), mask=m_rev_cur)

        # Update carry with sum of this tile
        carry += tl.sum(x_rev, axis=0)

        # Prefetch next tile if any
        next_block_idx = block_idx - 1
        if next_block_idx >= 0:
            base_next = next_block_idx * BLOCK_N
            rev_cols_next = base_next + (BLOCK_N - 1 - i)
            m_rev_next = rev_cols_next < N
            x_rev = tl.load(row_x_ptr + rev_cols_next * stride_x_col, mask=m_rev_next, other=0.0).to(tl.float32)


def cumsum_reverse_npu(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if not hasattr(torch, "npu") or not x.is_npu:
        raise ValueError("cumsum_reverse_npu requires an Ascend NPU tensor input")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError("cumsum_reverse_npu supports float16 and float32 inputs only")
    if x.ndim == 0:
        raise ValueError("cumsum_reverse_npu requires at least one dimension")

    dim = dim if dim >= 0 else (x.ndim + dim)
    if dim < 0 or dim >= x.ndim:
        raise IndexError(f"dim={dim} is out of range for ndim={x.ndim}")

    x_perm = x.movedim(dim, -1).contiguous()
    N = x_perm.shape[-1]
    rows = x_perm.numel() // N
    if rows == 0:
        return torch.empty_like(x)

    x2 = x_perm.reshape(rows, N)
    y2 = torch.empty_like(x2)

    BLOCK_N = 512
    NUM_BLOCKS = triton.cdiv(N, BLOCK_N)
    _rcumsum_lastdim_kernel[(rows,)](
        x2,
        y2,
        rows,
        N,
        x2.stride(0),
        x2.stride(1),
        y2.stride(0),
        y2.stride(1),
        NUM_BLOCKS=NUM_BLOCKS,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=4,
    )
    return y2.reshape(x_perm.shape).movedim(-1, dim)


class ModelNew(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim=1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return cumsum_reverse_npu(x, dim=self.dim)
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]
def get_init_inputs():
    return [dim]
