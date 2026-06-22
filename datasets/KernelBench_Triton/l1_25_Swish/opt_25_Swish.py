import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_BLOCK_SIZE = 8192
_MAX_PROGRAMS = 65535


@triton.jit
def _swish_direct_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    y = xf * tl.sigmoid(xf)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _swish_persistent_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    n_programs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.multiple_of(offs, 16)
        tl.max_contiguous(offs, 16)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        y = xf * tl.sigmoid(xf)
        tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """Swish activation: y = x * sigmoid(x)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-tracked inputs")

        x_contig = x.contiguous()
        n_elements = x_contig.numel()
        if n_elements == 0:
            return x_contig

        y = torch.empty_like(x_contig)
        n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
        if n_tiles > _MAX_PROGRAMS:
            n_programs = _MAX_PROGRAMS
            _swish_persistent_kernel[(n_programs, )](
                x_contig,
                y,
                n_elements,
                n_programs,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=8,
                num_stages=2,
            )
        else:
            _swish_direct_kernel[(n_tiles, )](
                x_contig,
                y,
                n_elements,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=8,
                num_stages=2,
            )
        return y


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
