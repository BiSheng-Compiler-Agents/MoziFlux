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

_MAX_PROGRAMS = 65535


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


@triton.jit
def _instancenorm_divide_direct_kernel(
    x_ptr,
    N,
    C,
    H,
    W,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    eps,
    div_const,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C
    offs = tl.arange(0, BLOCK_HW)
    hw = H * W
    mask = offs < hw
    base = n * stride_n + c * stride_c
    ptrs = x_ptr + base + offs

    vals = tl.load(ptrs, mask=mask, other=0.0)
    vals_f32 = vals.to(tl.float32)
    sum_x = tl.sum(vals_f32, axis=0)
    sum_x2 = tl.sum(vals_f32 * vals_f32, axis=0)

    inv_hw = tl.full((), 1.0 / hw, tl.float32)
    mean = sum_x * inv_hw
    var = tl.maximum(sum_x2 * inv_hw - mean * mean, 0.0)
    scale = tl.rsqrt(var + eps) * (1.0 / div_const)
    bias = -mean * scale
    out = tl.fma(vals_f32, scale, bias)
    tl.store(ptrs, out.to(vals.dtype), mask=mask)


@triton.jit
def _instancenorm_divide_persistent_kernel(
    x_ptr,
    total_rows,
    hw,
    eps,
    inv_hw,
    inv_div_const,
    n_programs,
    BLOCK_HW: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < hw
    while row < total_rows:
        ptrs = x_ptr + (row * hw + offs).to(tl.int64)
        vals = tl.load(ptrs, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sum_x = tl.sum(vals_f32, axis=0)
        sum_x2 = tl.sum(vals_f32 * vals_f32, axis=0)

        mean = sum_x * inv_hw
        var = tl.maximum(sum_x2 * inv_hw - mean * mean, 0.0)
        scale = tl.rsqrt(var + eps) * inv_div_const
        bias = -mean * scale
        out = tl.fma(vals_f32, scale, bias)
        tl.store(ptrs, out.to(vals.dtype), mask=mask)
        row += n_programs


class ModelNew(nn.Module):
    """Conv2d followed by InstanceNorm2d(no affine/running stats) and divide-by-constant."""

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        divide_by=DEFAULT_DIVIDE_BY,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")

        x = self.conv(x)
        if x.dtype not in (torch.float16, torch.float32):
            raise RuntimeError(
                "ModelNew supports only float16 and float32 inputs")

        x = x.contiguous()
        N, C, H, W = x.shape
        stride_n, stride_c, stride_h, stride_w = x.stride()
        hw = H * W
        total_rows = N * C
        block_hw = _next_power_of_two(hw)
        eps = float(self.instance_norm.eps)
        div_const = float(self.divide_by)

        if total_rows > _MAX_PROGRAMS or x.numel() > 2_000_000_000:
            n_programs = min(total_rows, _MAX_PROGRAMS)
            _instancenorm_divide_persistent_kernel[(n_programs, )](
                x,
                total_rows,
                hw,
                eps,
                1.0 / float(hw),
                1.0 / div_const,
                n_programs,
                BLOCK_HW=block_hw,
                num_warps=8,
                num_stages=2,
            )
        else:
            _instancenorm_divide_direct_kernel[(total_rows, )](
                x,
                N,
                C,
                H,
                W,
                stride_n,
                stride_c,
                stride_h,
                stride_w,
                eps,
                div_const,
                BLOCK_HW=block_hw,
                num_warps=8,
                num_stages=4,
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
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
divide_by = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="npu")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
