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

_MAX_GRID = 65535
_BLOCK_SIZE = 2048
_MODEL_CACHE = {}


@triton.jit
def _post_ops_channel_direct(
    x_ptr,
    bias_ptr,
    y_ptr,
    DHW: tl.constexpr,
    C: tl.constexpr,
    tiles_per_segment: tl.constexpr,
    total_tiles,
    bias_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    seg = tile_id // tiles_per_segment
    tile_in_seg = tile_id - seg * tiles_per_segment
    hw = tile_in_seg * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (tile_id < total_tiles) & (hw < DHW)
    base = seg * DHW + hw

    tl.multiple_of(hw, BLOCK_SIZE)
    tl.max_contiguous(hw, BLOCK_SIZE)

    x = tl.load(x_ptr + base, mask=mask, other=0.0,
                care_padding=False).to(tl.float32)
    x = tl.maximum(x, 0.0)
    # LeakyReLU immediately after ReLU is an exact no-op for non-negative x.
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))

    c = (seg - (seg // C) * C).to(tl.int32)
    b = tl.load(bias_ptr + c * bias_stride_c).to(tl.float32)
    tl.store(y_ptr + base, x + b, mask=mask)


@triton.jit
def _post_ops_channel_persistent(
    x_ptr,
    bias_ptr,
    y_ptr,
    DHW: tl.constexpr,
    C: tl.constexpr,
    tiles_per_segment: tl.constexpr,
    total_tiles,
    bias_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    for tile_id in tl.range(pid, total_tiles, n_programs, num_stages=2):
        seg = tile_id // tiles_per_segment
        tile_in_seg = tile_id - seg * tiles_per_segment
        hw = tile_in_seg * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = hw < DHW
        base = seg * DHW + hw

        tl.multiple_of(hw, BLOCK_SIZE)
        tl.max_contiguous(hw, BLOCK_SIZE)

        x = tl.load(x_ptr + base, mask=mask, other=0.0,
                    care_padding=False).to(tl.float32)
        x = tl.maximum(x, 0.0)
        # LeakyReLU immediately after ReLU is an exact no-op for non-negative x.
        x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        x = 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))

        c = (seg - (seg // C) * C).to(tl.int32)
        b = tl.load(bias_ptr + c * bias_stride_c).to(tl.float32)
        tl.store(y_ptr + base, x + b, mask=mask)


class ModelNew(nn.Module):
    """Conv3d followed by fused ReLU -> LeakyReLU -> GELU -> Sigmoid -> bias add."""

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super(ModelNew, self).__init__()
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
        n, c, d, h, w = y.shape
        n_elements = y.numel()
        if n_elements == 0:
            return y

        dhw = d * h * w
        b = self.bias.contiguous()
        tiles_per_segment = triton.cdiv(dhw, _BLOCK_SIZE)
        total_tiles = (n * c) * tiles_per_segment
        grid_n = min(total_tiles, _MAX_GRID)

        if total_tiles <= _MAX_GRID:
            _post_ops_channel_direct[(total_tiles, )](
                y,
                b,
                y,
                dhw,
                c,
                tiles_per_segment,
                total_tiles,
                b.stride(0),
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
        else:
            _post_ops_channel_persistent[(grid_n, )](
                y,
                b,
                y,
                dhw,
                c,
                tiles_per_segment,
                total_tiles,
                b.stride(0),
                BLOCK_SIZE=_BLOCK_SIZE,
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
