import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 8
DEFAULT_OUT_CHANNELS = 64
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_KERNEL_SIZE = 3
INIT_SEED = 2026

_MAX_GRID = 65535
_SMALL_BLOCK = 2048
_MEDIUM_BLOCK = 4096
_LARGE_BLOCK = 8192


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _hswish_relu_direct_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)

    # ReLU(HardSwish(x)) == 0 for x <= 0, else x * min(x + 3, 6) / 6.
    # This avoids materializing rx = max(x, 0) and removes one vector max chain.
    y_pos = x * tl.minimum(x + 3.0, 6.0) * (1.0 / 6.0)
    y = tl.where(x > 0.0, y_pos, 0.0)

    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _hswish_relu_persistent_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    n_programs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, n_tiles, n_programs, num_stages=2):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.multiple_of(offsets, 16)
        tl.max_contiguous(offsets, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        y_pos = x * tl.minimum(x + 3.0, 6.0) * (1.0 / 6.0)
        y = tl.where(x > 0.0, y_pos, 0.0)
        tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """Conv2d followed by fused ReLU(HardSwish(.)) epilogue."""

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def _select_block(self, n_elements: int) -> int:
        if n_elements >= (1 << 20):
            return _LARGE_BLOCK
        if n_elements >= (1 << 18):
            return _MEDIUM_BLOCK
        return _SMALL_BLOCK

    def _fused_hardswish_relu_triton(self, x: torch.Tensor) -> torch.Tensor:
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects Ascend NPU tensors for the Triton path")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(
                f"ModelNew supports float16, bfloat16, and float32 inputs, got {x.dtype}"
            )

        x_in = x.contiguous()
        n_elements = x_in.numel()
        if n_elements == 0:
            return torch.empty_like(x_in)

        block_size = self._select_block(n_elements)
        n_tiles = triton.cdiv(n_elements, block_size)
        y = torch.empty_like(x_in)

        if n_tiles > _MAX_GRID:
            n_programs = _MAX_GRID
            _hswish_relu_persistent_kernel[(n_programs,)](
                x_in,
                y,
                n_elements,
                n_programs,
                BLOCK_SIZE=block_size,
                num_warps=8,
                num_stages=2,
            )
        else:
            _hswish_relu_direct_kernel[(n_tiles,)](
                x_in,
                y,
                n_elements,
                BLOCK_SIZE=block_size,
                num_warps=8 if block_size >= _MEDIUM_BLOCK else 4,
                num_stages=2,
            )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects an Ascend NPU tensor input")
        x = self.conv(x)
        return self._fused_hardswish_relu_triton(x)


batch_size = DEFAULT_BATCH_SIZE
in_channels = DEFAULT_IN_CHANNELS
out_channels = DEFAULT_OUT_CHANNELS
height, width = DEFAULT_HEIGHT, DEFAULT_WIDTH
kernel_size = DEFAULT_KERNEL_SIZE


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
