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
SPECIAL_CASE_NUMEL = 128 * 64 * 126 * 126
SPECIAL_CASE_BLOCK = 9216
SPECIAL_CASE_WARPS = 4
SPECIAL_CASE_STAGES = 2


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _hswish_relu_masked_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    rx = tl.maximum(x, 0.0)
    y = rx * tl.minimum(rx * (1.0 / 6.0) + 0.5, 1.0)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _hswish_relu_exact_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.max_contiguous(tl.multiple_of(offsets, 16), BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    rx = tl.maximum(x, 0.0)
    y = rx * tl.minimum(rx * (1.0 / 6.0) + 0.5, 1.0)
    tl.store(y_ptr + offsets, y)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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

        y = torch.empty_like(x_in)
        if (
            n_elements == SPECIAL_CASE_NUMEL
            and x_in.shape == (DEFAULT_BATCH_SIZE, DEFAULT_OUT_CHANNELS, DEFAULT_HEIGHT - DEFAULT_KERNEL_SIZE + 1, DEFAULT_WIDTH - DEFAULT_KERNEL_SIZE + 1)
            and x_in.dtype == torch.float16
        ):
            grid = (SPECIAL_CASE_NUMEL // SPECIAL_CASE_BLOCK,)
            _hswish_relu_exact_kernel[grid](
                x_in,
                y,
                BLOCK_SIZE=SPECIAL_CASE_BLOCK,
                num_warps=SPECIAL_CASE_WARPS,
                num_stages=SPECIAL_CASE_STAGES,
            )
            return y

        if n_elements >= (1 << 20):
            block_size = 8192
            num_warps = 8
        elif n_elements >= (1 << 18):
            block_size = 4096
            num_warps = 8
        else:
            block_size = 2048
            num_warps = 4

        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _hswish_relu_masked_kernel[grid](
            x_in,
            y,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects an Ascend NPU tensor input")
        x = self.conv(x)
        x = self._fused_hardswish_relu_triton(x)
        return x


batch_size = DEFAULT_BATCH_SIZE
in_channels = DEFAULT_IN_CHANNELS
out_channels = DEFAULT_OUT_CHANNELS
height, width = DEFAULT_HEIGHT, DEFAULT_WIDTH
kernel_size = DEFAULT_KERNEL_SIZE
_MODEL_CACHE: dict[tuple[int | None, torch.dtype], ModelNew] = {}
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
