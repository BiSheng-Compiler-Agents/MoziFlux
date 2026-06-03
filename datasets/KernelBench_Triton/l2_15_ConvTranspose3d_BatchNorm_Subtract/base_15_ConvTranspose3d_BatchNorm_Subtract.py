import torch
import torch.nn as nn
import torch_npu  # noqa: F401

import triton
import triton.language as tl


DEFAULT_BATCH_SIZE = 16
DEFAULT_IN_CHANNELS = 16
DEFAULT_OUT_CHANNELS = 32
DEFAULT_DEPTH = 16
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 1


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048, "ROWS_PER_PROG": 4}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _spatial_mean_subtract_kernel(
    x_ptr,       # *: [N*C, S] contiguous in row-major layout
    y_ptr,       # *: [N*C, S] output
    stride_row,  # stride between flattened (n, c) rows
    S,           # total spatial elements per (n, c) = D*H*W
    TOTAL_ROWS,  # total flattened (n, c) rows = N*C
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offsets = pid * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)
    row_mask = row_offsets < TOTAL_ROWS
    base_x = x_ptr + row_offsets * stride_row
    base_y = y_ptr + row_offsets * stride_row

    offsets = tl.arange(0, BLOCK_SIZE)
    tl.static_assert(BLOCK_SIZE % 128 == 0)

    # Batch multiple rows together to amortize per-program setup and reduce launch count.
    sum_acc = tl.zeros((ROWS_PER_PROG,), dtype=tl.float32)
    i = 0
    while i < S:
        idx = i + offsets
        mask = row_mask[:, None] & (idx[None, :] < S)
        ptrs = base_x[:, None] + idx[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0, eviction_policy="evict_last")
        sum_acc += tl.sum(vals.to(tl.float32), axis=1)
        i += BLOCK_SIZE

    mean = sum_acc / tl.full((ROWS_PER_PROG,), S, dtype=tl.float32)

    i = 0
    while i < S:
        idx = i + offsets
        mask = row_mask[:, None] & (idx[None, :] < S)
        ptrs = base_x[:, None] + idx[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0, eviction_policy="evict_last")
        out = vals.to(tl.float32) - mean[:, None]
        tl.store(base_y[:, None] + idx[None, :], out.to(vals.dtype), mask=mask)
        i += BLOCK_SIZE


class ModelNew(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
        bias=True,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")

        x = self.conv_transpose(x)
        x = self.batch_norm(x)

        x_contig = x.contiguous()
        N, C, D, H, W = x_contig.shape
        S = D * H * W
        total_rows = N * C
        x_rows = x_contig.reshape(total_rows, S)
        row_stride = x_rows.stride(0)
        y = torch.empty_like(x_contig)
        y_rows = y.reshape(total_rows, S)

        grid = lambda meta: (triton.cdiv(total_rows, meta["ROWS_PER_PROG"]),)
        _spatial_mean_subtract_kernel[grid](
            x_rows,
            y_rows,
            row_stride,
            S,
            total_rows,
        )
        return y


_MODEL_CACHE: dict[tuple[torch.device, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    return model(x)
batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device='npu')]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
