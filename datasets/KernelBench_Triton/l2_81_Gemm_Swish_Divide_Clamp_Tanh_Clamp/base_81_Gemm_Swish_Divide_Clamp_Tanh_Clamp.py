import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_epilogue_swish_div_clamp_tanh(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base = pid * BLOCK
    offsets = base + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    sigmoid = tl.sigmoid(xf)
    y = xf * sigmoid
    y = y * 0.5
    y = tl.maximum(y, -1.0)
    y = tl.minimum(y, 1.0)
    e2y = tl.exp(2.0 * y)
    y = (e2y - 1.0) / (e2y + 1.0)
    y = tl.maximum(y, -1.0)
    y = tl.minimum(y, 1.0)

    tl.store(y_ptr + offsets, y.to(x.dtype), mask=mask)


@triton.jit
def _fused_epilogue_swish_div_clamp_tanh_nomask(
    x_ptr,
    y_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    x = tl.load(x_ptr + offsets)
    xf = x.to(tl.float32)

    sigmoid = tl.sigmoid(xf)
    y = xf * sigmoid
    y = y * 0.5
    y = tl.maximum(y, -1.0)
    y = tl.minimum(y, 1.0)
    e2y = tl.exp(2.0 * y)
    y = (e2y - 1.0) / (e2y + 1.0)
    y = tl.maximum(y, -1.0)
    y = tl.minimum(y, 1.0)

    tl.store(y_ptr + offsets, y.to(x.dtype))


def _run_fused_epilogue(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("Triton epilogue requires NPU tensors")
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    if x.ndim == 2 and x.shape == (1024, 8192):
        _fused_epilogue_swish_div_clamp_tanh_nomask[(n_elements // 4096,)](
            x,
            out,
            BLOCK=4096,
            num_warps=8,
            num_stages=3,
        )
        return out
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _fused_epilogue_swish_div_clamp_tanh[grid](
        x,
        out,
        n_elements,
        BLOCK=4096,
        num_warps=8,
        num_stages=2,
    )
    return out


def gemm_swish_divide_clamp_tanh_clamp(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    if x.device.type != "npu" or weight.device.type != "npu":
        raise RuntimeError("gemm_swish_divide_clamp_tanh_clamp requires NPU tensors")
    if bias is not None and bias.device.type != "npu":
        raise RuntimeError("bias must be an NPU tensor when provided")
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("x and weight must both be rank-2 tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must match weight.shape[1]")

    y = torch.matmul(x, weight.transpose(0, 1))
    if bias is not None:
        y = y + bias
    return _run_fused_epilogue(y)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return gemm_swish_divide_clamp_tanh_clamp(
            x, self.gemm.weight, self.gemm.bias
        )


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
