import torch
import torch.nn as nn
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 64
DEFAULT_IN_CHANNELS = 8
DEFAULT_OUT_CHANNELS = 32
DEFAULT_DEPTH = 32
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_BIAS_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1, 1)
_MODEL_CACHE = {}


@triton.jit
def _fused_post_ops_bias_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    n_elements,
    C,
    stride_c,
    bias_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    program_count = tl.num_programs(axis=0)
    tile_range = tl.arange(0, BLOCK_SIZE)
    total_programs = tl.cdiv(n_elements, BLOCK_SIZE)
    current = pid
    while current < total_programs:
        block_start = current * BLOCK_SIZE
        offs = block_start + tile_range
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.maximum(x, 0.0)
        inv_sqrt2 = 0.7071067811865476
        erf_term = tl.math.erf(x * inv_sqrt2)
        x = (x * (1.0 + erf_term)) * 0.5
        x = 1.0 / (1.0 + tl.exp(-x))
        c_idx = ((offs // stride_c) % C).to(tl.int32)
        bias = tl.load(bias_ptr + c_idx * bias_stride_c, mask=mask,
                       other=0.0).to(tl.float32)
        tl.store(y_ptr + offs, x + bias, mask=mask)
        current += program_count


@triton.jit
def _fused_post_ops_bias_slice_grid_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    n_elements,
    C,
    stride_c,
    bias_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    slice_idx = tl.program_id(axis=0)
    inner_tile = tl.program_id(axis=1)
    slice_stride = tl.num_programs(axis=0)
    tile_range = tl.arange(0, BLOCK_SIZE)
    while slice_idx * stride_c < n_elements:
        c_idx = (slice_idx % C).to(tl.int32)
        block_start = slice_idx * stride_c + inner_tile * BLOCK_SIZE
        offs = block_start + tile_range
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.maximum(x, 0.0)
        inv_sqrt2 = 0.7071067811865476
        erf_term = tl.math.erf(x * inv_sqrt2)
        x = (x * (1.0 + erf_term)) * 0.5
        x = 1.0 / (1.0 + tl.exp(-x))
        bias = tl.load(bias_ptr + c_idx * bias_stride_c).to(tl.float32)
        tl.store(y_ptr + offs, x + bias, mask=mask)
        slice_idx += slice_stride


class ModelNew(nn.Module):

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects an Ascend NPU tensor input")
        if self.conv.weight.device != x.device or self.conv.weight.dtype != x.dtype:
            self.conv = self.conv.to(device=x.device, dtype=x.dtype)
        if self.bias.device != x.device or self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(device=x.device, dtype=x.dtype)
        y = self.conv(x).contiguous()
        _, c, _, _, _ = y.shape
        n_elements = y.numel()
        if n_elements == 0:
            return y
        bias = self.bias.contiguous()
        stride_c = y.stride(1)
        bias_stride_c = bias.stride(0)
        block = 3720
        if stride_c % block == 0:
            channel_slices = n_elements // stride_c
            tiles_per_channel = stride_c // block
            slice_programs = min(channel_slices,
                                 max(1, 65535 // tiles_per_channel))
            grid = (slice_programs, tiles_per_channel)
            _fused_post_ops_bias_slice_grid_kernel[grid](
                y,
                bias,
                y,
                n_elements,
                c,
                stride_c,
                bias_stride_c,
                BLOCK_SIZE=block,
                num_warps=4,
                num_stages=2,
            )
        else:
            total_programs = triton.cdiv(n_elements, block)
            grid = (min(total_programs, 65535), )
            _fused_post_ops_bias_kernel[grid](
                y,
                bias,
                y,
                n_elements,
                c,
                stride_c,
                bias_stride_c,
                BLOCK_SIZE=block,
                num_warps=4,
                num_stages=2,
            )
        return y


def _set_deterministic_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def conv3d_relu_leakyrelu_gelu_sigmoid_biasadd(
        x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError(
            "conv3d_relu_leakyrelu_gelu_sigmoid_biasadd expects an Ascend NPU tensor"
        )
    key = (str(x.device), x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        _set_deterministic_seed(0)
        model = ModelNew(*get_init_inputs()).eval().to(device=x.device,
                                                       dtype=x.dtype)
        _MODEL_CACHE[key] = model
    with torch.no_grad():
        return model(x)


batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
