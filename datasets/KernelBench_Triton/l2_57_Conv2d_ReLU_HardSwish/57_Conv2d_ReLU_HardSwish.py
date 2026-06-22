import torch
import torch.nn as nn
import triton
import triton.language as tl

_MODEL_CACHE = {}


@triton.jit
def _relu_hswish_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hint for better vectorization on contiguous ranges
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Fused ReLU + HardSwish:
    # r = max(x, 0)
    # y = r * clamp((r + 3)/6, 0, 1)
    # For r >= 0, clamp reduces to min((r + 3)/6, 1) = min(r + 3, 6) * (1/6)
    r = tl.maximum(x, 0.0)
    inv6 = 1.0 / 6.0
    y = r * tl.minimum(r + 3.0, 6.0) * inv6

    tl.store(x_ptr + offsets, y, mask=mask)


def fused_relu_hardswish(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("fused_relu_hardswish expects an Ascend NPU tensor")

    x = x.contiguous()
    n_elements = x.numel()
    if n_elements == 0:
        return x

    if n_elements >= (1 << 22):
        block_size = 8192
        num_warps = 8
        num_stages = 2
    else:
        block_size = 4096
        num_warps = 4
        num_stages = 1

    grid = (triton.cdiv(n_elements, block_size), )
    _relu_hswish_inplace_kernel[grid](
        x,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return x


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and applies HardSwish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return fused_relu_hardswish(x)


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3


def _set_deterministic_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def conv2d_relu_hardswish(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError(
            "conv2d_relu_hardswish expects an Ascend NPU tensor")

    key = (str(x.device), x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        _set_deterministic_seed(0)
        model = ModelNew(*get_init_inputs()).eval().to(device=x.device,
                                                       dtype=x.dtype)
        _MODEL_CACHE[key] = model

    with torch.no_grad():
        return model(x)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
