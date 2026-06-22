import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_BLOCK_SIZE = 8192
_MAX_PROGRAMS = 65535


@triton.jit
def _elu_direct_kernel(x_ptr, y_ptr, n_elements, alpha,
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
    x_neg = tl.minimum(x, 0.0)
    inv_ln2 = 1.4426950408889634
    exp_term = tl.exp2(x_neg * inv_ln2)
    y = tl.where(x > 0.0, x, alpha * (exp_term - 1.0))
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _elu_persistent_kernel(x_ptr, y_ptr, n_elements, alpha, n_programs,
                           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        base = tile_id.to(tl.int64) * BLOCK_SIZE
        offsets = base + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        tl.multiple_of(offsets, 16)
        tl.max_contiguous(offsets, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_first")
        x_neg = tl.minimum(x, 0.0)
        inv_ln2 = 1.4426950408889634
        exp_term = tl.exp2(x_neg * inv_ln2)
        y = tl.where(x > 0.0, x, alpha * (exp_term - 1.0))
        tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """ELU activation optimized for large Ascend NPU tensors."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-tracked inputs")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.numel() == 0:
            return x.contiguous()

        x_contiguous = x.contiguous()
        y = torch.empty_like(x_contiguous)
        n_elements = x_contiguous.numel()
        n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)

        x_flat = x_contiguous.reshape(-1)
        y_flat = y.reshape(-1)
        if n_tiles > _MAX_PROGRAMS:
            _elu_persistent_kernel[(_MAX_PROGRAMS, )](
                x_flat,
                y_flat,
                n_elements,
                self.alpha,
                _MAX_PROGRAMS,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
        else:
            _elu_direct_kernel[(n_tiles, )](
                x_flat,
                y_flat,
                n_elements,
                self.alpha,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
        return y.reshape_as(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="npu")
    return [x]


def get_init_inputs():
    return [1.0]
