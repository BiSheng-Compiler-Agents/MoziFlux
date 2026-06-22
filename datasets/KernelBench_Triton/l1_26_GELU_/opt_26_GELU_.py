import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535  # Ascend FFTS grid cap


@triton.jit
def _gelu_fwd_kernel(
    x_ptr,
    y_ptr,
    rows,
    cols,
    row_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """GELU forward kernel — general path with boundary_check."""
    pid_row = tl.program_id(axis=0)
    row_start = pid_row * BLOCK_ROWS
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(rows, cols),
        strides=(row_stride, 1),
        offsets=(row_start, 0),
        block_shape=(BLOCK_ROWS, BLOCK_COLS),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(rows, cols),
        strides=(row_stride, 1),
        offsets=(row_start, 0),
        block_shape=(BLOCK_ROWS, BLOCK_COLS),
        order=(1, 0),
    )

    for _ in range(0, cols, BLOCK_COLS):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        x32 = x.to(tl.float32)
        x2 = x32 * x32
        inner = x32 * (0.7978845608028654 + 0.035677408136300125 * x2)
        tanh_inner = tl.math.tanh(inner)
        y = (0.5 * x32 * (1.0 + tanh_inner)).to(x.dtype)
        tl.store(y_block_ptr, y, boundary_check=(0, 1))
        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_COLS))
        y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_COLS))


@triton.jit
def _gelu_fwd_kernel_even(
    x_ptr,
    y_ptr,
    rows,
    cols,
    row_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """GELU forward kernel — fast path for power-of-2 shapes (no boundary_check)."""
    pid_row = tl.program_id(axis=0)
    row_start = pid_row * BLOCK_ROWS
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(rows, cols),
        strides=(row_stride, 1),
        offsets=(row_start, 0),
        block_shape=(BLOCK_ROWS, BLOCK_COLS),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(rows, cols),
        strides=(row_stride, 1),
        offsets=(row_start, 0),
        block_shape=(BLOCK_ROWS, BLOCK_COLS),
        order=(1, 0),
    )

    for _ in range(0, cols, BLOCK_COLS):
        x = tl.load(x_block_ptr)
        x32 = x.to(tl.float32)
        x2 = x32 * x32
        inner = x32 * (0.7978845608028654 + 0.035677408136300125 * x2)
        tanh_inner = tl.math.tanh(inner)
        y = (0.5 * x32 * (1.0 + tanh_inner)).to(x.dtype)
        tl.store(y_block_ptr, y)
        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_COLS))
        y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_COLS))


@triton.jit
def _gelu_fwd_kernel_persistent(
    x_ptr,
    y_ptr,
    rows,
    cols,
    row_stride,
    n_programs,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """GELU forward kernel — persistent work-stealing for shapes exceeding FFTS cap."""
    pid = tl.program_id(axis=0)
    num_row_blocks = tl.cdiv(rows, BLOCK_ROWS)
    for block_idx in range(pid, num_row_blocks, n_programs):
        row_start = block_idx * BLOCK_ROWS
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(rows, cols),
            strides=(row_stride, 1),
            offsets=(row_start, 0),
            block_shape=(BLOCK_ROWS, BLOCK_COLS),
            order=(1, 0),
        )
        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(rows, cols),
            strides=(row_stride, 1),
            offsets=(row_start, 0),
            block_shape=(BLOCK_ROWS, BLOCK_COLS),
            order=(1, 0),
        )
        for _ in range(0, cols, BLOCK_COLS):
            x = tl.load(x_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero")
            x32 = x.to(tl.float32)
            x2 = x32 * x32
            inner = x32 * (0.7978845608028654 + 0.035677408136300125 * x2)
            tanh_inner = tl.math.tanh(inner)
            y = (0.5 * x32 * (1.0 + tanh_inner)).to(x.dtype)
            tl.store(y_block_ptr, y, boundary_check=(0, 1))
            x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_COLS))
            y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_COLS))


class ModelNew(nn.Module):
    """
    Optimized GELU activation using a Triton kernel on Ascend NPU tensors.
    Uses two-path dispatch (direct + persistent), tl.math.tanh, and
    2D blocking with make_block_ptr for efficient boundary handling.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        supported_dtypes = {torch.float16, torch.float32, torch.bfloat16}
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-tracked inputs")

        x_contig = x.contiguous()
        if x_contig.numel() == 0:
            return x_contig

        cols = x_contig.shape[-1]
        rows = x_contig.numel() // cols
        y = torch.empty_like(x_contig)

        BLOCK_ROWS = 4
        BLOCK_COLS = 2048

        n_tiles = triton.cdiv(rows, BLOCK_ROWS)
        is_even = (rows % BLOCK_ROWS == 0) and (cols % BLOCK_COLS == 0)

        if n_tiles > _MAX_PROGRAMS:
            kernel = _gelu_fwd_kernel_persistent
            grid = (_MAX_PROGRAMS, )
            kernel[grid](
                x_contig.view(-1),
                y.view(-1),
                rows,
                cols,
                cols,
                _MAX_PROGRAMS,
                BLOCK_ROWS=BLOCK_ROWS,
                BLOCK_COLS=BLOCK_COLS,
            )
        elif is_even:
            kernel = _gelu_fwd_kernel_even
            grid = (n_tiles, )
            kernel[grid](
                x_contig.view(-1),
                y.view(-1),
                rows,
                cols,
                cols,
                BLOCK_ROWS=BLOCK_ROWS,
                BLOCK_COLS=BLOCK_COLS,
            )
        else:
            kernel = _gelu_fwd_kernel
            grid = (n_tiles, )
            kernel[grid](
                x_contig.view(-1),
                y.view(-1),
                rows,
                cols,
                cols,
                BLOCK_ROWS=BLOCK_ROWS,
                BLOCK_COLS=BLOCK_COLS,
            )

        return y


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="npu")
    return [x]


def get_init_inputs():
    return []
