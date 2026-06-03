import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _relu_divide_inplace_kernel(x_ptr, n_elements, inv_divisor, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x > 0, x, 0.0) * inv_divisor
    tl.store(x_ptr + offsets, x, mask=mask)


def _require_npu_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "npu":
        raise RuntimeError(f"{name} must be on NPU, got {tensor.device}")


def gemm_relu_divide(x, weight, bias, divisor):
    _require_npu_tensor("x", x)
    _require_npu_tensor("weight", weight)
    _require_npu_tensor("bias", bias)
    if x.device != weight.device or x.device != bias.device:
        raise RuntimeError("x, weight, and bias must be on the same NPU device")

    divisor = float(divisor)
    if divisor == 0.0:
        raise ValueError("divisor must be non-zero")

    inv_divisor = 1.0 / divisor

    y = F.linear(x, weight, bias).contiguous()
    n_elements = y.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _relu_divide_inplace_kernel[grid](
        y,
        n_elements,
        inv_divisor,
        BLOCK_SIZE=16192,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    def __init__(
        self,
        in_features=1024,
        out_features=512,
        divisor=2.0,
        *,
        dtype=torch.float32,
        device="npu",
    ):
        super().__init__()
        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.divisor = float(divisor)

    def forward(self, x):
        return gemm_relu_divide(x, self.linear.weight, self.linear.bias, self.divisor)
batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]
def get_init_inputs():
    return [in_features, out_features, divisor]
