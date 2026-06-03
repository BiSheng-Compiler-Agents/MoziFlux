import torch
import torch_npu  # noqa: F401
import torch.nn as nn
import triton
import triton.language as tl

TARGET_BATCH = 32768
TARGET_COLS = 32768
TARGET_ROWS = TARGET_BATCH - 1
FAST_BLOCK_M = 4
FAST_BLOCK_N = 2048
FAST_NUM_CHUNKS = TARGET_COLS // FAST_BLOCK_N
FAST_NUM_WARPS = 4
FAST_NUM_STAGES = 2


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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_M
    rows_idx = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows_idx < rows
    cols = tl.arange(0, BLOCK_N)

    zero_col_ptrs = y_ptr + rows_idx * stride_y_row
    tl.store(zero_col_ptrs, 0.0, mask=row_mask)

    carry = tl.zeros((BLOCK_M,), dtype=tl.float32)
    x_row_ptrs = x_ptr + rows_idx[:, None] * stride_x_row
    y_row_ptrs = y_ptr + rows_idx[:, None] * stride_y_row

    for chunk_idx in tl.static_range(NUM_CHUNKS):
        chunk_cols = chunk_idx * BLOCK_N + cols
        mask = row_mask[:, None] & (chunk_cols[None, :] < n_cols_in)
        x_ptrs = x_row_ptrs + chunk_cols[None, :] * stride_x_col
        y_ptrs = y_row_ptrs + (chunk_cols[None, :] + 1) * stride_y_col

        vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        prefix = tl.cumsum(vals, axis=1)
        outputs = prefix + carry[:, None]
        tl.store(y_ptrs, outputs, mask=mask)
        carry += tl.sum(vals, axis=1)


@triton.jit
def _exclusive_cumsum_row_to_padded_fast_kernel(
    x_ptr,
    y_ptr,
    rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_M
    rows_idx = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows_idx < rows
    cols = tl.arange(0, BLOCK_N)

    zero_col_ptrs = y_ptr + rows_idx * 32769
    tl.store(zero_col_ptrs, 0.0, mask=row_mask)

    carry = tl.zeros((BLOCK_M,), dtype=tl.float32)
    x_row_ptrs = x_ptr + rows_idx[:, None] * 32768
    y_row_ptrs = y_ptr + rows_idx[:, None] * 32769

    for chunk_idx in tl.static_range(NUM_CHUNKS):
        chunk_cols = chunk_idx * BLOCK_N + cols
        x_ptrs = x_row_ptrs + chunk_cols[None, :]
        y_ptrs = y_row_ptrs + (chunk_cols[None, :] + 1)

        vals = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        prefix = tl.cumsum(vals, axis=1)
        outputs = prefix + carry[:, None]
        tl.store(y_ptrs, outputs, mask=row_mask[:, None])
        carry += tl.sum(vals, axis=1)


class ModelNew(nn.Module):
    def __init__(self, dim=1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        if not x.is_npu:
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.ndim != 2:
            raise RuntimeError(
                f"ModelNew expects a 2D tensor, but received ndim={x.ndim}"
            )
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

        rows = B - 1
        y = torch.empty((rows, N + 1), device=x.device, dtype=x.dtype)
        stride_x_row, stride_x_col = x.stride()
        stride_y_row, stride_y_col = y.stride()

        use_fast_path = (
            self.dim == 1
            and x.dtype == torch.float16
            and B == TARGET_BATCH
            and N == TARGET_COLS
            and stride_x_row == TARGET_COLS
            and stride_x_col == 1
            and stride_y_row == TARGET_COLS + 1
            and stride_y_col == 1
        )
        if use_fast_path:
            grid = (triton.cdiv(rows, FAST_BLOCK_M),)
            _exclusive_cumsum_row_to_padded_fast_kernel[grid](
                x[:rows],
                y,
                rows,
                BLOCK_M=FAST_BLOCK_M,
                BLOCK_N=FAST_BLOCK_N,
                NUM_CHUNKS=FAST_NUM_CHUNKS,
                num_warps=FAST_NUM_WARPS,
                num_stages=FAST_NUM_STAGES,
            )
            return y

        if N <= 128:
            block_n = 128
            block_m = 16
        elif N <= 512:
            block_n = 256
            block_m = 8
        elif N <= 2048:
            block_n = 512
            block_m = 4
        else:
            block_n = 1024
            block_m = 2

        grid = (triton.cdiv(rows, block_m),)
        _exclusive_cumsum_row_to_padded_kernel[grid](
            x[:rows],
            y,
            rows,
            N,
            stride_x_row,
            stride_x_col,
            stride_y_row,
            stride_y_col,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            NUM_CHUNKS=triton.cdiv(N, block_n),
            num_warps=4,
            num_stages=2,
        )
        return y


batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    return [torch.rand(batch_size, *input_shape, device="npu")]


def get_init_inputs():
    return [dim]
