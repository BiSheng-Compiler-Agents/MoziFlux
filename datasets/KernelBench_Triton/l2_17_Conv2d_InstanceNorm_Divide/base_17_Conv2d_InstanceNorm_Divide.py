import math
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 128
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DIVIDE_BY = 2.0


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _instancenorm_divide_2d_fused_kernel(
    x_ptr,  # input/output
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    eps, div_const,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # each program handles one (n, c)
    n = pid // C
    c = pid % C

    # Flattened spatial index [0, H*W)
    offs = tl.arange(0, BLOCK_HW)
    hw = H * W
    mask = offs < hw

    # Because the input is made contiguous by the caller, spatial slice is contiguous in memory.
    base = n * stride_n + c * stride_c
    ptrs = x_ptr + base + offs

    # Load values (masked), accumulate sum and sumsq in fp32
    x_vals = tl.load(ptrs, mask=mask, other=0.0)
    x_f32 = x_vals.to(tl.float32)

    sum_x = tl.sum(x_f32, axis=0)
    sum_x2 = tl.sum(x_f32 * x_f32, axis=0)

    # Compute mean and variance (population variance)
    inv_hw = tl.full((), 1.0 / hw, tl.float32)
    mean = sum_x * inv_hw
    var = sum_x2 * inv_hw - mean * mean
    var = tl.maximum(var, 0.0)

    inv_std = tl.rsqrt(var + eps)
    rcp_div = 1.0 / div_const
    scale = inv_std * rcp_div
    bias = -mean * scale

    # Normalize and divide-by in one pass; store back
    y = tl.fma(x_f32, scale, bias)
    tl.store(ptrs, y.to(x_vals.dtype), mask=mask)


@triton.jit
def _instancenorm_divide_2d_hw15876_kernel(
    x_ptr,
    rows,
    row_stride,
    eps,
    div_const,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_HW)
    base = pid * row_stride
    ptrs = x_ptr + base + offs

    x_vals = tl.load(ptrs)
    x_f32 = x_vals.to(tl.float32)
    sum_x = tl.sum(x_f32, axis=0)
    sum_x2 = tl.sum(x_f32 * x_f32, axis=0)

    inv_hw = tl.full((), 1.0 / 15876.0, tl.float32)
    mean = sum_x * inv_hw
    var = sum_x2 * inv_hw - mean * mean
    var = tl.maximum(var, 0.0)

    inv_std = tl.rsqrt(var + eps)
    scale = inv_std * (1.0 / div_const)
    bias = -mean * scale
    y = tl.fma(x_f32, scale, bias)
    tl.store(ptrs, y.to(x_vals.dtype))


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    This version fuses InstanceNorm (no affine, no running stats) and the final division into a single
    Triton kernel for improved performance.
    """
    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        divide_by=DEFAULT_DIVIDE_BY,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Keep the module to mirror original semantics; use its eps for correctness.
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")

        x = self.conv(x)
        if x.dtype not in (torch.float16, torch.float32):
            raise RuntimeError("ModelNew supports only float16 and float32 inputs")

        N, C, H, W = x.shape
        # Ensure contiguous layout for predictable strides
        x = x.contiguous()
        stride_n, stride_c, stride_h, stride_w = x.stride()

        # Triton kernel: one program per (n, c), vectorize across H*W
        HW = H * W
        BLOCK_HW = _next_power_of_two(HW)
        grid = (N * C,)

        eps = float(self.instance_norm.eps)
        div_const = float(self.divide_by)

        if HW == 15876 and stride_w == 1:
            _instancenorm_divide_2d_hw15876_kernel[grid](
                x,
                N * C,
                stride_c,
                eps,
                div_const,
                BLOCK_HW=15876,
                num_warps=8,
                num_stages=4,
            )
        else:
            _instancenorm_divide_2d_fused_kernel[grid](
                x,  # in-place
                N, C, H, W,
                stride_n, stride_c, stride_h, stride_w,
                eps, div_const,
                BLOCK_HW=BLOCK_HW,
                num_warps=2,
                num_stages=1,
            )
        return x


_MODEL_CACHE: dict[tuple[torch.device, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    return model(x)
batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128  
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='npu')]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
