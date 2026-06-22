import torch
import torch.nn as nn
import triton
import triton.language as tl

_MODEL_CACHE = {}


@triton.jit
def _relu_hswish_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ASSIGN_CONTIG: tl.constexpr,
    USE_CONTIG_HINT: tl.constexpr,
    USE_MULTIPLE_OF: tl.constexpr,
    MATH_MODE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if ASSIGN_CONTIG:
        offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
    elif USE_CONTIG_HINT:
        tl.max_contiguous(offsets, BLOCK_SIZE)
    if USE_MULTIPLE_OF:
        tl.multiple_of(offsets, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    r = tl.maximum(x, 0.0)
    inv6 = 1.0 / 6.0
    if MATH_MODE == 0:
        y = r * tl.minimum(r + 3.0, 6.0) * inv6
    elif MATH_MODE == 1:
        y = r * tl.minimum(r * inv6 + 0.5, 1.0)
    else:
        scaled = (r + 3.0) * inv6
        y = tl.where(r < 3.0, r * scaled, r)
    tl.store(x_ptr + offsets, y, mask=mask)


def _resolve_launch_config(
        _n_elements: int) -> tuple[int, int, int, bool, bool, bool, int]:
    return (8192, 4, 1, True, False, False, 0)


def fused_relu_hardswish(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("fused_relu_hardswish expects an Ascend NPU tensor")

    x = x.contiguous()
    n_elements = x.numel()
    if n_elements == 0:
        return x

    block_size, num_warps, num_stages, assign_contig, use_contig_hint, use_multiple_of, math_mode = _resolve_launch_config(
        n_elements)
    grid = (triton.cdiv(n_elements, block_size), )
    _relu_hswish_inplace_kernel[grid](
        x,
        n_elements,
        BLOCK_SIZE=block_size,
        ASSIGN_CONTIG=assign_contig,
        USE_CONTIG_HINT=use_contig_hint,
        USE_MULTIPLE_OF=use_multiple_of,
        MATH_MODE=math_mode,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return x


class ModelNew(nn.Module):

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
