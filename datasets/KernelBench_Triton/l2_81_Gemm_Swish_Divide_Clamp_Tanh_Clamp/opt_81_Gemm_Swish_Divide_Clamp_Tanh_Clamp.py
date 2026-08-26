import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

_BLOCK_SIZE = 4096
_MAX_PROGRAMS = 65535


@triton.jit
def _epilogue_direct(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    xf = x.to(tl.float32)

    # swish(x) / 2, then hard clamp to the tanh input domain.
    y = (xf / (1.0 + tl.exp(-xf))) * 0.5
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    # tanh via exp formulation because tl.tanh is unavailable on triton-ascend.
    e2y = tl.exp(2.0 * y)
    y = (e2y - 1.0) / (e2y + 1.0)
    tl.store(y_ptr + offsets, y.to(x.dtype), mask=mask)


@triton.jit
def _epilogue_persistent(x_ptr, y_ptr, n_elements, n_programs,
                         BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in range(pid, n_tiles, n_programs):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        xf = x.to(tl.float32)
        y = (xf / (1.0 + tl.exp(-xf))) * 0.5
        y = tl.minimum(tl.maximum(y, -1.0), 1.0)
        e2y = tl.exp(2.0 * y)
        y = (e2y - 1.0) / (e2y + 1.0)
        tl.store(y_ptr + offsets, y.to(x.dtype), mask=mask)


def _run_fused_epilogue(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("Triton epilogue requires NPU tensors")
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
    if n_tiles > _MAX_PROGRAMS:
        _epilogue_persistent[(_MAX_PROGRAMS, )](x,
                                                out,
                                                n_elements,
                                                _MAX_PROGRAMS,
                                                BLOCK=_BLOCK_SIZE,
                                                num_warps=4,
                                                num_stages=2)
    else:
        _epilogue_direct[(n_tiles, )](x,
                                      out,
                                      n_elements,
                                      BLOCK=_BLOCK_SIZE,
                                      num_warps=4,
                                      num_stages=2)
    return out


def gemm_swish_divide_clamp_tanh_clamp(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None) -> torch.Tensor:
    if x.device.type != "npu" or weight.device.type != "npu":
        raise RuntimeError(
            "gemm_swish_divide_clamp_tanh_clamp requires NPU tensors")
    if bias is not None and bias.device.type != "npu":
        raise RuntimeError("bias must be an NPU tensor when provided")
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("x and weight must both be rank-2 tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must match weight.shape[1]")

    # F.linear lets the backend use its optimized GEMM+bias path and returns a contiguous epilogue input.
    y = F.linear(x, weight, bias)
    return _run_fused_epilogue(y)


class ModelNew(nn.Module):
    """GEMM followed by swish, divide, clamp, tanh, and final clamp (mathematically redundant after tanh)."""

    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return gemm_swish_divide_clamp_tanh_clamp(x, self.gemm.weight,
                                                  self.gemm.bias)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
