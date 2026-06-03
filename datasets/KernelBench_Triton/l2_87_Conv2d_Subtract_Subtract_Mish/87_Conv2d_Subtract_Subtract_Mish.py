import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 8
DEFAULT_OUT_CHANNELS = 64
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_KERNEL_SIZE = 3
DEFAULT_SUBTRACT_VALUE_1 = 0.5
DEFAULT_SUBTRACT_VALUE_2 = 0.2
INIT_SEED = 2026


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _fused_sub_mish_kernel(
    x_ptr,          # in-place pointer to tensor
    n_elements,     # total number of elements
    sub1,           # subtract_value_1 (scalar)
    sub2,           # subtract_value_2 (scalar)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load and upcast for numerics
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Apply sequential subtractions: (x - sub1) - sub2
    x32 = x32 - sub1
    x32 = x32 - sub2

    zero = tl.zeros_like(x32)
    one = zero + 1.0
    twenty = zero + 20.0
    neg_twenty = zero - 20.0

    # Match PyTorch softplus threshold behavior for better numerical parity.
    abs_x = tl.abs(x32)
    sp_mid = tl.where(x32 > zero, x32, zero) + tl.log(one + tl.exp(-abs_x))
    sp = tl.where(x32 > twenty, x32, tl.where(x32 < neg_twenty, tl.exp(x32), sp_mid))
    y32 = x32 * tl.tanh(sp)
    y = y32.to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


def _fused_sub_mish_inplace(x: torch.Tensor, sub1: float, sub2: float) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError("_fused_sub_mish_inplace expects an Ascend NPU tensor")
    if x.requires_grad:
        raise RuntimeError("_fused_sub_mish_inplace does not support autograd-tracked tensors")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "_fused_sub_mish_inplace supports only float16, bfloat16, and float32 inputs"
        )

    n_elements = x.numel()
    if n_elements == 0:
        return x
    x = x.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fused_sub_mish_kernel[grid](
        x,
        n_elements,
        float(sub1),
        float(sub2),
        BLOCK_SIZE=4096,
        num_warps=8,
        num_stages=1,
    )
    return x


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    """
    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        subtract_value_1: float = DEFAULT_SUBTRACT_VALUE_1,
        subtract_value_2: float = DEFAULT_SUBTRACT_VALUE_2,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = float(subtract_value_1)
        self.subtract_value_2 = float(subtract_value_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects an Ascend NPU tensor input")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")
        x = self.conv(x)
        return _fused_sub_mish_inplace(x, self.subtract_value_1, self.subtract_value_2)


batch_size = DEFAULT_BATCH_SIZE
in_channels = DEFAULT_IN_CHANNELS
out_channels = DEFAULT_OUT_CHANNELS
height, width = DEFAULT_HEIGHT, DEFAULT_WIDTH
kernel_size = DEFAULT_KERNEL_SIZE
subtract_value_1 = DEFAULT_SUBTRACT_VALUE_1
subtract_value_2 = DEFAULT_SUBTRACT_VALUE_2
_MODEL_CACHE: dict[tuple[int | None, torch.dtype], ModelNew] = {}
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    device="npu",
    dtype=torch.float32,
    return [torch.rand(batch_size, in_channels, height, width, device=device)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]