import torch
import torch_npu  # noqa: F401
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _exclusive_cumsum_row_to_padded_kernel(
    x_ptr,
    y_ptr,
    rows,
    n_cols_in,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    BLOCK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row
    tl.store(row_y_ptr, 0.0)

    running = tl.zeros((), dtype=tl.float32)

    for chunk_idx in tl.static_range(NUM_CHUNKS):
        col_start = chunk_idx * BLOCK_SIZE
        base_x = row_x_ptr + col_start * stride_x_col
        base_y = row_y_ptr + (col_start + 1) * stride_y_col

        px = base_x
        py = base_y
        for offset in tl.static_range(BLOCK_SIZE):
            col = col_start + offset
            in_bounds = col < n_cols_in
            value = tl.load(px, mask=in_bounds, other=0.0).to(tl.float32)
            running += value
            tl.store(py, running, mask=in_bounds)
            px += stride_x_col
            py += stride_y_col


class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim=1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        if not x.is_npu:
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.ndim != 2:
            raise RuntimeError(
                f"ModelNew expects a 2D tensor, but received ndim={x.ndim}")
        if (self.dim % x.ndim) != 1:
            raise RuntimeError(
                f"ModelNew only supports dim=1 or -1 for 2D inputs, but received dim={self.dim}"
            )
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}")

        x = x.contiguous()
        B, N = x.shape
        if N == 0:
            return x.new_empty((max(B - 1, 0), 1))
        if B <= 1:
            return x.new_empty((0, N + 1))

        y = torch.empty((B - 1, N + 1), device=x.device, dtype=x.dtype)

        if N <= 128:
            block = 128
        elif N <= 512:
            block = 256
        else:
            block = 512

        grid = (B - 1, )
        _exclusive_cumsum_row_to_padded_kernel[grid](
            x,
            y,
            B - 1,
            N,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_SIZE=block,
            NUM_CHUNKS=triton.cdiv(N, block),
            num_warps=4,
            num_stages=2,
        )
        return y


batch_size = 32768
input_shape = (32768, )
dim = 1


def get_inputs():
    return [torch.rand(batch_size, *input_shape, device='npu')]


def get_init_inputs():
    return [dim]
