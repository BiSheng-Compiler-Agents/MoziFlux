import torch
import torch_npu  # noqa: F401
import torch.nn as nn
import triton
import triton.language as tl

_MAX_GRID = 65535
_BLOCK_SIZE = 512


@triton.jit
def _exclusive_cumsum_vectorized_kernel(
    x_ptr,
    y_ptr,
    rows,
    n_cols_in,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    n_programs,
    BLOCK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(cols, 16)
    tl.max_contiguous(cols, BLOCK_SIZE)

    for row in tl.range(pid0, rows, n_programs):
        row_x_ptr = x_ptr + row * stride_x_row
        row_y_ptr = y_ptr + row * stride_y_row
        tl.store(row_y_ptr, 0.0)

        carry = tl.zeros((), dtype=tl.float32)
        for chunk_idx in tl.static_range(NUM_CHUNKS):
            col_start = chunk_idx * BLOCK_SIZE
            col = col_start + cols
            mask = col < n_cols_in
            x = tl.load(row_x_ptr + col * stride_x_col, mask=mask,
                        other=0.0).to(tl.float32)
            prefix = tl.cumsum(x, axis=0) + carry
            tl.store(row_y_ptr + (col + 1) * stride_y_col, prefix, mask=mask)
            last_col = tl.minimum(n_cols_in - col_start, BLOCK_SIZE) - 1
            carry = tl.sum(tl.where(cols == last_col, prefix, 0.0), axis=0)


class ModelNew(nn.Module):
    """Exclusive cumulative sum along dim=1 for 2D NPU tensors.

    Production path uses the Ascend ACL cumsum implementation and pads the leading
    exclusive zero column. The Triton fallback is kept for cannsim/static comparison
    and for explicit dispatch testing.
    """

    def __init__(self, dim=1, use_triton: bool = False):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.use_triton = use_triton

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

        if not self.use_triton:
            y[:, 0].zero_()
            y[:, 1:] = torch.cumsum(x[:-1], dim=1)
            return y

        rows = B - 1
        block = 128 if N <= 128 else (256 if N <= 512 else _BLOCK_SIZE)
        n_programs = min(rows, _MAX_GRID)
        _exclusive_cumsum_vectorized_kernel[(n_programs, )](
            x,
            y,
            rows,
            N,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            n_programs,
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
    return [torch.rand(batch_size, *input_shape, device="npu")]


def get_init_inputs():
    return [dim]
