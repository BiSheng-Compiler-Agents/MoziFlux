import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_KERNEL_SIZE = 8
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 4
_MAX_PROGRAMS = 65535


@triton.jit
def _avgpool1d_cols_kernel(
    x_ptr,
    y_ptr,
    L_IN,
    L_OUT,
    COL_BLOCK_START,
    STRIDE,
    PADDING,
    KERNEL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_block = COL_BLOCK_START + tl.program_id(1)
    block_offsets = tl.arange(0, BLOCK)
    offs = col_block * BLOCK + block_offsets
    mask_o = offs < L_OUT
    starts = offs * STRIDE - PADDING
    x_row = x_ptr + row_id * L_IN
    y_row = y_ptr + row_id * L_OUT
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(0, KERNEL_SIZE):
        pos = starts + k
        valid = (pos >= 0) & (pos < L_IN) & mask_o
        safe_pos = tl.minimum(tl.maximum(pos, 0), L_IN - 1)
        vals = tl.load(x_row + safe_pos, mask=valid, other=0.0)
        acc += vals.to(tl.float32)
    tl.store(y_row + offs, acc * (1.0 / float(KERNEL_SIZE)), mask=mask_o)


@triton.jit
def _avgpool1d_row_kernel(
    x_ptr,
    y_ptr,
    N_ROWS,
    L_IN,
    L_OUT,
    N_COL_BLOCKS: tl.constexpr,
    STRIDE,
    PADDING,
    N_PROGRAMS,
    KERNEL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_offsets = tl.arange(0, BLOCK)
    while row_id < N_ROWS:
        x_row = x_ptr + row_id * L_IN
        y_row = y_ptr + row_id * L_OUT
        for col_block in tl.range(0, N_COL_BLOCKS):
            offs = col_block * BLOCK + block_offsets
            mask_o = offs < L_OUT
            starts = offs * STRIDE - PADDING
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for k in tl.static_range(0, KERNEL_SIZE):
                pos = starts + k
                valid = (pos >= 0) & (pos < L_IN) & mask_o
                safe_pos = tl.minimum(tl.maximum(pos, 0), L_IN - 1)
                vals = tl.load(x_row + safe_pos, mask=valid, other=0.0)
                acc += vals.to(tl.float32)
            tl.store(y_row + offs,
                     acc * (1.0 / float(KERNEL_SIZE)),
                     mask=mask_o)
        row_id += N_PROGRAMS


class ModelNew(nn.Module):
    """1D average pooling with count_include_pad=True and ceil_mode=False."""

    def __init__(self,
                 kernel_size: int = DEFAULT_KERNEL_SIZE,
                 stride: int = DEFAULT_STRIDE,
                 padding: int = DEFAULT_PADDING):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise RuntimeError(
                "ModelNew supports float16, float32, and bfloat16 inputs only")
        if x.ndim != 3:
            raise RuntimeError(
                f"ModelNew expects [batch, channels, length], got {tuple(x.shape)}"
            )
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-tracked inputs")
        if self.kernel_size <= 0 or self.stride <= 0 or self.padding < 0:
            raise RuntimeError(
                "kernel_size and stride must be positive, and padding must be non-negative"
            )

        x = x.contiguous()
        B, C, L_in = x.shape
        L_out = (L_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        if L_out <= 0:
            return x.new_empty((B, C, 0))
        y = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)
        n_rows = B * C
        block = 256
        n_col_blocks = triton.cdiv(L_out, block)
        x2 = x.view(n_rows, L_in)
        y2 = y.view(n_rows, L_out)

        if n_rows <= _MAX_PROGRAMS:
            cols_per_launch = max(1, _MAX_PROGRAMS // max(1, n_rows))
            start = 0
            while start < n_col_blocks:
                cols = min(cols_per_launch, n_col_blocks - start)
                _avgpool1d_cols_kernel[(n_rows, cols)](
                    x2,
                    y2,
                    L_in,
                    L_out,
                    start,
                    self.stride,
                    self.padding,
                    KERNEL_SIZE=self.kernel_size,
                    BLOCK=block,
                    num_warps=4,
                    num_stages=2,
                )
                start += cols
        else:
            n_programs = _MAX_PROGRAMS
            _avgpool1d_row_kernel[(n_programs, )](
                x2,
                y2,
                n_rows,
                L_in,
                L_out,
                n_col_blocks,
                self.stride,
                self.padding,
                n_programs,
                KERNEL_SIZE=self.kernel_size,
                BLOCK=block,
                num_warps=4,
                num_stages=2,
            )
        return y


def avg_pool1d(x: torch.Tensor,
               kernel_size: int = DEFAULT_KERNEL_SIZE,
               stride: int = DEFAULT_STRIDE,
               padding: int = DEFAULT_PADDING) -> torch.Tensor:
    return ModelNew(kernel_size=kernel_size, stride=stride, padding=padding)(x)


batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4


def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]
