import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl


_EPILOGUE_BLOCK_SIZE = 4096
_MAX_PROGRAMS = 65535


@triton.jit
def _relu_scale_direct_kernel(x_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    y = tl.where(x > 0.0, x * scale, 0.0)
    tl.store(x_ptr + offsets, y, mask=mask)


@triton.jit
def _relu_scale_persistent_kernel(x_ptr, n_elements, n_programs, scale,
                                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.multiple_of(offsets, 16)
        tl.max_contiguous(offsets, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        y = tl.where(x > 0.0, x * scale, 0.0)
        tl.store(x_ptr + offsets, y, mask=mask)


def _require_npu_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "npu":
        raise RuntimeError(f"{name} must be on NPU, got {tensor.device}")


def _launch_relu_scale_inplace(y, scale):
    n_elements = y.numel()
    if n_elements == 0:
        return y
    n_tiles = triton.cdiv(n_elements, _EPILOGUE_BLOCK_SIZE)
    if n_tiles > _MAX_PROGRAMS:
        grid = (_MAX_PROGRAMS,)
        _relu_scale_persistent_kernel[grid](
            y,
            n_elements,
            _MAX_PROGRAMS,
            scale,
            BLOCK_SIZE=_EPILOGUE_BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    else:
        grid = (n_tiles,)
        _relu_scale_direct_kernel[grid](
            y,
            n_elements,
            scale,
            BLOCK_SIZE=_EPILOGUE_BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    return y


def gemm_relu_divide(x, weight, bias, divisor):
    _require_npu_tensor("x", x)
    _require_npu_tensor("weight", weight)
    _require_npu_tensor("bias", bias)
    if x.device != weight.device or x.device != bias.device:
        raise RuntimeError("x, weight, and bias must be on the same NPU device")

    divisor = float(divisor)
    if divisor == 0.0:
        raise ValueError("divisor must be non-zero")

    y = F.linear(x, weight, bias).contiguous()
    return _launch_relu_scale_inplace(y, 1.0 / divisor)


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
        return gemm_relu_divide(x, self.linear.weight, self.linear.bias,
                                self.divisor)


batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, divisor]
