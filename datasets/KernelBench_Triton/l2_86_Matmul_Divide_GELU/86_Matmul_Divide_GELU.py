import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 1024
DEFAULT_IN_FEATURES = 512
DEFAULT_OUT_FEATURES = 1024
DEFAULT_DIVISOR = 10.0


@triton.jit
def fused_div_gelu_kernel(x_ptr, out_ptr, n_elements, inv_divisor,
                          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    y = x32 * inv_divisor

    y3 = y * y * y
    inner = 0.7978845608028654 * (y + 0.044715 * y3)
    abs_inner = tl.abs(inner)
    exp_term = tl.exp(-2.0 * abs_inner)
    tanh_abs = (1.0 - exp_term) / (1.0 + exp_term)
    tanh_inner = tl.where(inner >= 0.0, tanh_abs, -tanh_abs)
    out = (0.5 * y * (1.0 + tanh_inner)).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


def _require_npu_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "npu":
        raise RuntimeError(
            f"{name} must be on Ascend NPU, got {tensor.device}")


def _select_launch_config(n_elements: int) -> tuple[int, int, int]:
    if n_elements <= 262_144:
        return 1024, 4, 2
    if n_elements <= (1 << 20):
        return 4096, 8, 2
    return 8192, 8, 2


def matmul_divide_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    divisor: float = DEFAULT_DIVISOR,
) -> torch.Tensor:
    _require_npu_tensor("x", x)
    _require_npu_tensor("weight", weight)
    if bias is not None:
        _require_npu_tensor("bias", bias)

    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            "matmul_divide_gelu expects x and weight to be 2D tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must match weight.shape[1]")
    if x.device != weight.device:
        raise RuntimeError("x and weight must be on the same NPU device")
    if x.dtype != weight.dtype:
        raise TypeError("x and weight must use the same dtype")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "matmul_divide_gelu supports float16, bfloat16, and float32")

    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(
                "bias must be a 1D tensor with shape [weight.shape[0]]")
        if bias.device != x.device:
            raise RuntimeError("bias must be on the same NPU device as x")
        if bias.dtype != x.dtype:
            raise TypeError("bias must use the same dtype as x")
        bias = bias.contiguous()

    divisor = float(divisor)
    if divisor == 0.0:
        raise ValueError("divisor must be non-zero")

    linear_out = F.linear(x.contiguous(), weight.contiguous(),
                          bias).contiguous()
    n_elements = linear_out.numel()
    out = torch.empty_like(linear_out)
    if n_elements == 0:
        return out

    block_size, num_warps, num_stages = _select_launch_config(n_elements)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    fused_div_gelu_kernel[grid](
        linear_out,
        out,
        n_elements,
        1.0 / divisor,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


class ModelNew(nn.Module):

    def __init__(
        self,
        input_size: int = DEFAULT_IN_FEATURES,
        output_size: int = DEFAULT_OUT_FEATURES,
        divisor: float = DEFAULT_DIVISOR,
        *,
        dtype: torch.dtype = torch.float16,
        device: str = "npu",
    ):
        super().__init__()
        self.linear = nn.Linear(
            input_size,
            output_size,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.divisor = float(divisor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight
        bias = self.linear.bias
        if weight.device != x.device or weight.dtype != x.dtype:
            weight = weight.to(device=x.device, dtype=x.dtype)
        if bias is not None and (bias.device != x.device
                                 or bias.dtype != x.dtype):
            bias = bias.to(device=x.device, dtype=x.dtype)
        return matmul_divide_gelu(x, weight, bias, self.divisor)


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, output_size, divisor]
