import torch
import torch.nn as nn
import torch_npu  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def _rcumsum_lastdim_kernel_bench_exact(
    x_ptr,
    y_ptr,
    rows,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row
    i = tl.arange(0, BLOCK_N)
    rev_i = (BLOCK_N - 1) - i
    carry = tl.zeros((), dtype=tl.float32)
    cols_3 = 24576 + i
    x_tile_3 = tl.load(row_x_ptr + cols_3 * stride_x_col).to(tl.float32)
    x_rev_3 = tl.gather(x_tile_3, rev_i, axis=0)
    scan_rev_3 = tl.cumsum(x_rev_3, axis=0)
    out_rev_3 = scan_rev_3
    out_fwd_3 = tl.gather(out_rev_3, rev_i, axis=0)
    tl.store(row_y_ptr + cols_3 * stride_y_col, out_fwd_3)
    carry += tl.sum(x_tile_3, axis=0)
    cols_2 = 16384 + i
    x_tile_2 = tl.load(row_x_ptr + cols_2 * stride_x_col).to(tl.float32)
    x_rev_2 = tl.gather(x_tile_2, rev_i, axis=0)
    scan_rev_2 = tl.cumsum(x_rev_2, axis=0)
    out_rev_2 = scan_rev_2 + carry
    out_fwd_2 = tl.gather(out_rev_2, rev_i, axis=0)
    tl.store(row_y_ptr + cols_2 * stride_y_col, out_fwd_2)
    carry += tl.sum(x_tile_2, axis=0)
    cols_1 = 8192 + i
    x_tile_1 = tl.load(row_x_ptr + cols_1 * stride_x_col).to(tl.float32)
    x_rev_1 = tl.gather(x_tile_1, rev_i, axis=0)
    scan_rev_1 = tl.cumsum(x_rev_1, axis=0)
    out_rev_1 = scan_rev_1 + carry
    out_fwd_1 = tl.gather(out_rev_1, rev_i, axis=0)
    tl.store(row_y_ptr + cols_1 * stride_y_col, out_fwd_1)
    carry += tl.sum(x_tile_1, axis=0)
    cols_0 = 0 + i
    x_tile_0 = tl.load(row_x_ptr + cols_0 * stride_x_col).to(tl.float32)
    x_rev_0 = tl.gather(x_tile_0, rev_i, axis=0)
    scan_rev_0 = tl.cumsum(x_rev_0, axis=0)
    out_rev_0 = scan_rev_0 + carry
    out_fwd_0 = tl.gather(out_rev_0, rev_i, axis=0)
    tl.store(row_y_ptr + cols_0 * stride_y_col, out_fwd_0)
    carry += tl.sum(x_tile_0, axis=0)


@triton.jit
def _rcumsum_lastdim_kernel_aligned(
    x_ptr,
    y_ptr,
    rows,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row
    i = tl.arange(0, BLOCK_N)
    rev_i = (BLOCK_N - 1) - i
    carry = tl.zeros((), dtype=tl.float32)

    last_block = NUM_BLOCKS - 1
    base = last_block * BLOCK_N
    cols = base + i
    x_tile = tl.load(row_x_ptr + cols * stride_x_col).to(tl.float32)

    for b in range(NUM_BLOCKS):
        block_idx = NUM_BLOCKS - 1 - b
        base_cur = block_idx * BLOCK_N
        cols_cur = base_cur + i

        x_rev = tl.gather(x_tile, rev_i, axis=0)
        scan_rev = tl.cumsum(x_rev, axis=0)
        out_rev = scan_rev + carry
        out_fwd = tl.gather(out_rev, rev_i, axis=0)
        tl.store(row_y_ptr + cols_cur * stride_y_col, out_fwd)
        carry += tl.sum(x_tile, axis=0)

        next_block_idx = block_idx - 1
        if next_block_idx >= 0:
            base_next = next_block_idx * BLOCK_N
            cols_next = base_next + i
            x_tile = tl.load(row_x_ptr + cols_next * stride_x_col).to(
                tl.float32)


@triton.jit
def _rcumsum_lastdim_kernel_masked(
    x_ptr,
    y_ptr,
    rows,
    N,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row
    i = tl.arange(0, BLOCK_N)
    rev_i = (BLOCK_N - 1) - i
    carry = tl.zeros((), dtype=tl.float32)

    last_block = NUM_BLOCKS - 1
    base = last_block * BLOCK_N
    cols = base + i
    mask = cols < N
    x_tile = tl.load(row_x_ptr + cols * stride_x_col, mask=mask,
                     other=0.0).to(tl.float32)

    for b in range(NUM_BLOCKS):
        block_idx = NUM_BLOCKS - 1 - b
        base_cur = block_idx * BLOCK_N
        cols_cur = base_cur + i
        mask_cur = cols_cur < N

        x_rev = tl.gather(x_tile, rev_i, axis=0)
        scan_rev = tl.cumsum(x_rev, axis=0)
        out_rev = scan_rev + carry
        out_fwd = tl.gather(out_rev, rev_i, axis=0)
        tl.store(row_y_ptr + cols_cur * stride_y_col, out_fwd, mask=mask_cur)
        carry += tl.sum(x_tile, axis=0)

        next_block_idx = block_idx - 1
        if next_block_idx >= 0:
            base_next = next_block_idx * BLOCK_N
            cols_next = base_next + i
            mask_next = cols_next < N
            x_tile = tl.load(
                row_x_ptr + cols_next * stride_x_col,
                mask=mask_next,
                other=0.0,
            ).to(tl.float32)


def cumsum_reverse_npu(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if not hasattr(torch, "npu") or not x.is_npu:
        raise ValueError(
            "cumsum_reverse_npu requires an Ascend NPU tensor input")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(
            "cumsum_reverse_npu supports float16 and float32 inputs only")
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

    BLOCK_N = 8192
    NUM_BLOCKS = triton.cdiv(N, BLOCK_N)
    if N == 32768:
        _rcumsum_lastdim_kernel_bench_exact[(rows, )](
            x2,
            y2,
            rows,
            x2.stride(0),
            x2.stride(1),
            y2.stride(0),
            y2.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=4,
        )
    elif N % BLOCK_N == 0:
        _rcumsum_lastdim_kernel_aligned[(rows, )](
            x2,
            y2,
            rows,
            x2.stride(0),
            x2.stride(1),
            y2.stride(0),
            y2.stride(1),
            NUM_BLOCKS=NUM_BLOCKS,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=4,
        )
    else:
        _rcumsum_lastdim_kernel_masked[(rows, )](
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

    def __init__(self, dim=1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return cumsum_reverse_npu(x, dim=self.dim)


batch_size = 32768
input_shape = (32768, )
dim = 1


def get_inputs():
    return [torch.rand(batch_size, *input_shape)]


def get_init_inputs():
    return [dim]
