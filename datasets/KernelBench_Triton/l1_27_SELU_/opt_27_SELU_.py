import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_SELU_SCALE = 1.0507009873554805
_SELU_ALPHA = 1.6732632423543772
_BLOCK_SIZE = 8192
_MAX_PROGRAMS = 65535


@triton.jit
def _selu_direct_kernel(x_ptr, y_ptr, n_elements, ALPHA: tl.constexpr,
                        SCALE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
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
    neg = tl.minimum(x32, 0.0)
    pos = tl.maximum(x32, 0.0)
    out = pos * SCALE + (tl.exp(neg) - 1.0) * (SCALE * ALPHA)
    tl.store(y_ptr + offsets, out.to(x.dtype), mask=mask)


@triton.jit
def _selu_persistent_kernel(x_ptr, y_ptr, n_elements, n_programs,
                            ALPHA: tl.constexpr, SCALE: tl.constexpr,
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
        neg = tl.minimum(x32, 0.0)
        pos = tl.maximum(x32, 0.0)
        out = pos * SCALE + (tl.exp(neg) - 1.0) * (SCALE * ALPHA)
        tl.store(y_ptr + offsets, out.to(x.dtype), mask=mask)


class ModelNew(nn.Module):
    """SELU activation optimized for Ascend NPU tensors."""

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
            n_programs = _MAX_PROGRAMS
            _selu_persistent_kernel[(n_programs, )](
                x_contig,
                y,
                n_elements,
                n_programs,
                ALPHA=_SELU_ALPHA,
                SCALE=_SELU_SCALE,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
        else:
            _selu_direct_kernel[(n_tiles, )](
                x_contig,
                y,
                n_elements,
                ALPHA=_SELU_ALPHA,
                SCALE=_SELU_SCALE,
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
