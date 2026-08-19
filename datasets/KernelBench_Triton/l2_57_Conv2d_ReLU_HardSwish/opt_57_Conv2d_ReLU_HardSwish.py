import torch
import torch.nn as nn
import triton
import triton.language as tl

_BLOCK_SIZE = 8192
_MAX_PROGRAMS = 65535
_MODEL_CACHE = {}


@triton.jit
def _relu_hswish_direct_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    y_pos = x * tl.minimum(x + 3.0, 6.0) * (1.0 / 6.0)
    y = tl.where(x > 0.0, y_pos, 0.0)
    tl.store(x_ptr + offsets, y, mask=mask)


@triton.jit
def _relu_hswish_persistent_kernel(x_ptr, n_elements, n_programs,
                                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        base = tile_id.to(tl.int64) * BLOCK_SIZE
        offsets = base + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        y_pos = x * tl.minimum(x + 3.0, 6.0) * (1.0 / 6.0)
        y = tl.where(x > 0.0, y_pos, 0.0)
        tl.store(x_ptr + offsets, y, mask=mask)


def fused_relu_hardswish(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("fused_relu_hardswish expects an Ascend NPU tensor")
    x = x.contiguous()
    n_elements = x.numel()
    if n_elements == 0:
        return x
    n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
    if n_tiles > _MAX_PROGRAMS:
        _relu_hswish_persistent_kernel[(_MAX_PROGRAMS, )](
            x,
            n_elements,
            _MAX_PROGRAMS,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
            num_stages=2)
    else:
        _relu_hswish_direct_kernel[(n_tiles, )](x,
                                                n_elements,
                                                BLOCK_SIZE=_BLOCK_SIZE,
                                                num_warps=8,
                                                num_stages=2)
    return x


class ModelNew(nn.Module):
    """Conv2d followed by fused ReLU + HardSwish epilogue."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return fused_relu_hardswish(x)


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
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


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
