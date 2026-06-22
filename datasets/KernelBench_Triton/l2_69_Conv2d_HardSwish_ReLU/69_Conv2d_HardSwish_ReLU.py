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


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _hswish_relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1D launch over the flattened tensor
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Provide vectorization hints for better memory coalescing
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute ReLU(HardSwish(x)) with minimal ops:
    # rx = max(x, 0)
    # r  = min(rx/6 + 0.5, 1)
    # y  = rx * r
    rx = tl.maximum(x, 0.0)
    y = rx * tl.minimum(rx * (1.0 / 6.0) + 0.5, 1.0)

    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU.
    """

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
            raise RuntimeError(
                "ModelNew expects Ascend NPU tensors for the Triton path")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(
                f"ModelNew supports float16, bfloat16, and float32 inputs, got {x.dtype}"
            )

        x_in = x.contiguous()
        n_elements = x_in.numel()
        if n_elements == 0:
            return torch.empty_like(x_in)

        if n_elements >= (1 << 20):
            BLOCK_SIZE = 8192
            num_warps = 8
        elif n_elements >= (1 << 18):
            BLOCK_SIZE = 4096
            num_warps = 8
        else:
            BLOCK_SIZE = 2048
            num_warps = 4

        def grid(META):
            return (triton.cdiv(n_elements, META["BLOCK_SIZE"]), )

        y = torch.empty_like(x_in)
        _hswish_relu_kernel[grid](
            x_in,
            y,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=1,
        )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height - kernel_size + 1, width - kernel_size + 1).
        """
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
