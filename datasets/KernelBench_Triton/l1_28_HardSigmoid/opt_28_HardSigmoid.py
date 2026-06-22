import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_BLOCK_SIZE = 8192
_MAX_PROGRAMS = 65535


@triton.jit
def _hardsigmoid_direct_kernel(x_ptr, y_ptr, n_elements,
                               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first")
    x32 = x.to(tl.float32)
    y_mid = x32 * (1.0 / 6.0) + 0.5
    y32 = tl.where(x32 <= -3.0, 0.0, y_mid)
    y32 = tl.where(x32 >= 3.0, 1.0, y32)
    tl.store(y_ptr + offsets, y32.to(x.dtype), mask=mask)


@triton.jit
def _hardsigmoid_persistent_kernel(x_ptr, y_ptr, n_elements, n_programs,
                                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.multiple_of(offsets, 16)
        tl.max_contiguous(offsets, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_first")
        x32 = x.to(tl.float32)
        y_mid = x32 * (1.0 / 6.0) + 0.5
        y32 = tl.where(x32 <= -3.0, 0.0, y_mid)
        y32 = tl.where(x32 >= 3.0, 1.0, y32)
        tl.store(y_ptr + offsets, y32.to(x.dtype), mask=mask)


class ModelNew(nn.Module):
    """HardSigmoid activation optimized for large Ascend NPU tensors."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-tracked inputs")
        if x.numel() == 0:
            return x.contiguous()

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        n_elements = x_contig.numel()
        n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)

        if n_tiles > _MAX_PROGRAMS:
            _hardsigmoid_persistent_kernel[(_MAX_PROGRAMS, )](
                x_contig,
                y,
                n_elements,
                _MAX_PROGRAMS,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
        else:
            _hardsigmoid_direct_kernel[(n_tiles, )](
                x_contig,
                y,
                n_elements,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
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
