import torch
import torch.nn as nn
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK_ROWS = 4
_BLOCK_COLS = 2048


@triton.jit
def _gelu_fwd_kernel(x_ptr, y_ptr, rows, cols, row_stride,
                     BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    pid_row = tl.program_id(0)
    row_start = pid_row * BLOCK_ROWS
    x_block_ptr = tl.make_block_ptr(base=x_ptr,
                                    shape=(rows, cols),
                                    strides=(row_stride, 1),
                                    offsets=(row_start, 0),
                                    block_shape=(BLOCK_ROWS, BLOCK_COLS),
                                    order=(1, 0))
    y_block_ptr = tl.make_block_ptr(base=y_ptr,
                                    shape=(rows, cols),
                                    strides=(row_stride, 1),
                                    offsets=(row_start, 0),
                                    block_shape=(BLOCK_ROWS, BLOCK_COLS),
                                    order=(1, 0))
    for _ in range(0, cols, BLOCK_COLS):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        x32 = x.to(tl.float32)
        x2 = x32 * x32
        u = x32 * (0.7978845608028654 + 0.035677408136300125 * x2)
        y = (0.5 * x32 * (1.0 + tl.math.tanh(u))).to(x.dtype)
        tl.store(y_block_ptr, y, boundary_check=(0, 1))
        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_COLS))
        y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_COLS))


@triton.jit
def _gelu_fwd_kernel_even(x_ptr, y_ptr, rows, cols, row_stride,
                          BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    pid_row = tl.program_id(0)
    row_start = pid_row * BLOCK_ROWS
    x_block_ptr = tl.make_block_ptr(base=x_ptr,
                                    shape=(rows, cols),
                                    strides=(row_stride, 1),
                                    offsets=(row_start, 0),
                                    block_shape=(BLOCK_ROWS, BLOCK_COLS),
                                    order=(1, 0))
    y_block_ptr = tl.make_block_ptr(base=y_ptr,
                                    shape=(rows, cols),
                                    strides=(row_stride, 1),
                                    offsets=(row_start, 0),
                                    block_shape=(BLOCK_ROWS, BLOCK_COLS),
                                    order=(1, 0))
    for _ in range(0, cols, BLOCK_COLS):
        x = tl.load(x_block_ptr)
        x32 = x.to(tl.float32)
        x2 = x32 * x32
        u = x32 * (0.7978845608028654 + 0.035677408136300125 * x2)
        y = (0.5 * x32 * (1.0 + tl.math.tanh(u))).to(x.dtype)
        tl.store(y_block_ptr, y)
        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_COLS))
        y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_COLS))


@triton.jit
def _gelu_fwd_kernel_persistent(x_ptr, y_ptr, rows, cols, row_stride,
                                n_programs, BLOCK_ROWS: tl.constexpr,
                                BLOCK_COLS: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    n_row_tiles = tl.cdiv(rows, BLOCK_ROWS)
    for tile in range(pid, n_row_tiles, n_programs):
        row_start = tile * BLOCK_ROWS
        x_block_ptr = tl.make_block_ptr(base=x_ptr,
                                        shape=(rows, cols),
                                        strides=(row_stride, 1),
                                        offsets=(row_start, 0),
                                        block_shape=(BLOCK_ROWS, BLOCK_COLS),
                                        order=(1, 0))
        y_block_ptr = tl.make_block_ptr(base=y_ptr,
                                        shape=(rows, cols),
                                        strides=(row_stride, 1),
                                        offsets=(row_start, 0),
                                        block_shape=(BLOCK_ROWS, BLOCK_COLS),
                                        order=(1, 0))
        for _ in range(0, cols, BLOCK_COLS):
            if EVEN:
                x = tl.load(x_block_ptr)
            else:
                x = tl.load(x_block_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero")
            x32 = x.to(tl.float32)
            x2 = x32 * x32
            u = x32 * (0.7978845608028654 + 0.035677408136300125 * x2)
            y = (0.5 * x32 * (1.0 + tl.math.tanh(u))).to(x.dtype)
            if EVEN:
                tl.store(y_block_ptr, y)
            else:
                tl.store(y_block_ptr, y, boundary_check=(0, 1))
            x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_COLS))
            y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_COLS))


def _gelu_tanh_triton(x: torch.Tensor) -> torch.Tensor:
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    if x_contig.numel() == 0:
        return y.view_as(x)
    cols = x_contig.shape[-1]
    rows = x_contig.numel() // cols
    row_tiles = triton.cdiv(rows, _BLOCK_ROWS)
    even = (rows % _BLOCK_ROWS == 0) and (cols % _BLOCK_COLS == 0)
    if row_tiles > _MAX_PROGRAMS:
        n_programs = _MAX_PROGRAMS
        _gelu_fwd_kernel_persistent[(n_programs, )](x_contig.view(-1),
                                                    y.view(-1),
                                                    rows,
                                                    cols,
                                                    cols,
                                                    n_programs,
                                                    BLOCK_ROWS=_BLOCK_ROWS,
                                                    BLOCK_COLS=_BLOCK_COLS,
                                                    EVEN=even,
                                                    num_warps=4,
                                                    num_stages=2)
    else:
        kernel = _gelu_fwd_kernel_even if even else _gelu_fwd_kernel
        kernel[(row_tiles, )](x_contig.view(-1),
                              y.view(-1),
                              rows,
                              cols,
                              cols,
                              BLOCK_ROWS=_BLOCK_ROWS,
                              BLOCK_COLS=_BLOCK_COLS,
                              num_warps=4,
                              num_stages=2)
    return y.view_as(x)


class ModelNew(nn.Module):
    """Approximate GELU used by OpenAI GPT / minGPT."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU.")
        return _gelu_tanh_triton(x)


batch_size = 8192
dim = 8192


def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
