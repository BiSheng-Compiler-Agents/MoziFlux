import torch
import torch.nn as nn
import triton
import triton.language as tl

_DIRECT_BLOCK_SIZE = 4096
_PERSISTENT_BLOCK_SIZE = 8192
_MAX_PROGRAMS = 65535  # Ascend FFTS 1D grid cap


@triton.jit
def _hardtanh_direct_kernel(x_ptr, y_ptr, n_elements, min_val, max_val,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    # Preserve PyTorch NaN behavior: comparisons with NaN are false, so NaN flows through.
    y = tl.where(x > max_val, max_val, tl.where(x < min_val, min_val, x))
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _hardtanh_persistent_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    n_programs,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        base = tile_id.to(tl.int64) * BLOCK_SIZE
        offsets = base + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        # Preserve PyTorch NaN behavior: comparisons with NaN are false, so NaN flows through.
        y = tl.where(x > max_val, max_val, tl.where(x < min_val, min_val, x))
        tl.store(y_ptr + offsets, y, mask=mask)


def _hardtanh_triton(x: torch.Tensor,
                     min_val: float = -1.0,
                     max_val: float = 1.0) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return y

    direct_tiles = triton.cdiv(n_elements, _DIRECT_BLOCK_SIZE)
    if direct_tiles > _MAX_PROGRAMS:
        triton.cdiv(n_elements, _PERSISTENT_BLOCK_SIZE)
        n_programs = _MAX_PROGRAMS
        _hardtanh_persistent_kernel[(n_programs, )](
            x,
            y,
            n_elements,
            n_programs,
            min_val,
            max_val,
            BLOCK_SIZE=_PERSISTENT_BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    else:
        _hardtanh_direct_kernel[(direct_tiles, )](
            x,
            y,
            n_elements,
            min_val,
            max_val,
            BLOCK_SIZE=_DIRECT_BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    return y


class ModelNew(nn.Module):
    """HardTanh activation backed by direct/persistent Triton kernels on Ascend NPU."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(x, "is_npu") or not x.is_npu:
            raise RuntimeError(
                "ModelNew expects an input tensor on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")
        return _hardtanh_triton(x, -1.0, 1.0)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
