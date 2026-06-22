import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_BLOCK_SIZE = 8192
_MAX_PROGRAMS = 65535


@triton.jit
def _softsign_direct_kernel(x_ptr, y_ptr, n_elements,
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
    denom = tl.abs(x) + 1.0
    y = x / denom
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _softsign_persistent_kernel(x_ptr, y_ptr, n_elements, n_programs,
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
        denom = tl.abs(x) + 1.0
        y = x / denom
        tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """Softsign activation optimized for large Ascend NPU tensors."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-tracked inputs")
        if x.numel() == 0:
            return x.contiguous()

        x_contiguous = x.contiguous()
        y = torch.empty_like(x_contiguous)
        n_elements = x_contiguous.numel()
        n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)

        if n_tiles > _MAX_PROGRAMS:
            _softsign_persistent_kernel[(_MAX_PROGRAMS, )](
                x_contiguous.reshape(-1),
                y.reshape(-1),
                n_elements,
                _MAX_PROGRAMS,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
        else:
            _softsign_direct_kernel[(n_tiles, )](
                x_contiguous.reshape(-1),
                y.reshape(-1),
                n_elements,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
        return y.reshape_as(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device='npu')
    return [x]


def get_init_inputs():
    return []
