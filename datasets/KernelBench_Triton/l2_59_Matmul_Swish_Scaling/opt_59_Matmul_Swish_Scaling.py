import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_FEATURES = 32768
DEFAULT_OUT_FEATURES = 32768
DEFAULT_SCALING_FACTOR = 2.0

_SMALL_BLOCK_SIZE = 1024
_MEDIUM_BLOCK_SIZE = 4096
_LARGE_BLOCK_SIZE = 8192
_MAX_PROGRAMS = 65535


@triton.jit
def _swish_scale_direct(x_ptr, y_ptr, n_elements, scale,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_SIZE)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                care_padding=False).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid * scale
    tl.store(y_ptr + offs, out, mask=mask)


@triton.jit
def _swish_scale_persistent(x_ptr, y_ptr, n_elements, n_programs, scale,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        tl.multiple_of(offs, 16)
        tl.max_contiguous(offs, BLOCK_SIZE)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                    care_padding=False).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-x))
        out = x * sigmoid * scale
        tl.store(y_ptr + offs, out, mask=mask)


def swish_scale_triton(x: torch.Tensor, scale: float) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("swish_scale_triton requires Ascend NPU tensors.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "swish_scale_triton supports float16, bfloat16, and float32.")

    x_c = x.contiguous()
    y = torch.empty_like(x_c)
    n_elements = x_c.numel()
    if n_elements == 0:
        return y

    if n_elements <= 262_144:
        block_size = _SMALL_BLOCK_SIZE
    elif n_elements <= (1 << 20):
        block_size = _MEDIUM_BLOCK_SIZE
    else:
        block_size = _LARGE_BLOCK_SIZE

    n_tiles = triton.cdiv(n_elements, block_size)
    scale_scalar = float(scale)
    if n_tiles > _MAX_PROGRAMS:
        _swish_scale_persistent[(_MAX_PROGRAMS, )](
            x_c,
            y,
            n_elements,
            _MAX_PROGRAMS,
            scale_scalar,
            BLOCK_SIZE=block_size,
            num_warps=8,
            num_stages=2,
        )
    else:
        _swish_scale_direct[(n_tiles, )](
            x_c,
            y,
            n_elements,
            scale_scalar,
            BLOCK_SIZE=block_size,
            num_warps=8,
            num_stages=2,
        )
    return y


def matmul_swish_scaling(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: float = DEFAULT_SCALING_FACTOR,
) -> torch.Tensor:
    if x.device.type != "npu" or weight.device.type != "npu":
        raise RuntimeError("matmul_swish_scaling requires Ascend NPU tensors.")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            "matmul_swish_scaling expects x and weight to be 2D tensors.")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must match weight.shape[1].")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device.")
    if x.dtype != weight.dtype:
        raise ValueError("x and weight must use the same dtype.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "matmul_swish_scaling supports float16, bfloat16, and float32.")

    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(
                "bias must be a 1D tensor with shape [weight.shape[0]].")
        if bias.device != x.device:
            raise ValueError("bias must be on the same device as x.")
        if bias.dtype != x.dtype:
            raise ValueError("bias must use the same dtype as x.")
        bias = bias.contiguous()

    linear_out = F.linear(x.contiguous(), weight.contiguous(), bias)
    return swish_scale_triton(linear_out, scale)


class ModelNew(nn.Module):
    """Matrix multiplication followed by Swish activation and scaling."""

    def __init__(
        self,
        in_features: int = DEFAULT_IN_FEATURES,
        out_features: int = DEFAULT_OUT_FEATURES,
        scaling_factor: float = DEFAULT_SCALING_FACTOR,
    ):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        weight = self.matmul.weight.to(device=x.device, dtype=x.dtype)
        bias = None if self.matmul.bias is None else self.matmul.bias.to(
            device=x.device, dtype=x.dtype)
        return matmul_swish_scaling(x, weight, bias, self.scaling_factor)


batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor]
