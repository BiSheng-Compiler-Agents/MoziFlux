import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 128
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_KERNEL_SIZE = 3
DEFAULT_CONSTANT_VALUE = 0.5
DEFAULT_BIAS_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1)
DEFAULT_SCALING_FACTOR = 2.0

_BLOCK_HW = 8192
_MAX_PROGRAMS = 65535


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _min_bias_scale_direct_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    total_tiles,
    hw,
    channels,
    const_value,
    scaling,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_hw_tiles = tl.cdiv(hw, BLOCK_HW)
    plane = pid // n_hw_tiles
    hw_tile = pid - plane * n_hw_tiles
    hw_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = (pid < total_tiles) & (hw_offsets < hw)

    x_offsets = plane * hw + hw_offsets
    c_idx = plane % channels
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_idx, mask=pid < total_tiles, other=0.0)
    out = (tl.minimum(x, const_value) + bias) * scaling
    tl.store(out_ptr + x_offsets, out, mask=mask)


@triton.jit
def _min_bias_scale_persistent_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    total_tiles,
    hw,
    channels,
    n_programs,
    const_value,
    scaling,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_hw_tiles = tl.cdiv(hw, BLOCK_HW)
    for tile_id in range(pid, total_tiles, n_programs):
        plane = tile_id // n_hw_tiles
        hw_tile = tile_id - plane * n_hw_tiles
        hw_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = hw_offsets < hw
        x_offsets = plane * hw + hw_offsets
        c_idx = plane % channels
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + c_idx)
        out = (tl.minimum(x, const_value) + bias) * scaling
        tl.store(out_ptr + x_offsets, out, mask=mask)


def _apply_min_bias_scale(
    x: torch.Tensor,
    bias: torch.Tensor,
    constant_value: float,
    scaling_factor: float,
) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "_apply_min_bias_scale expects input tensors on Ascend NPU")
    if x.ndim != 4:
        raise RuntimeError(
            f"_apply_min_bias_scale expects a 4D tensor, got {tuple(x.shape)}")
    if x.numel() == 0:
        return x.clone()

    bias_1d = bias.reshape(-1)
    if bias_1d.numel() != x.shape[1]:
        raise RuntimeError(
            f"Bias channel count {bias_1d.numel()} does not match output channels {x.shape[1]}"
        )

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    hw = x_contig.shape[2] * x_contig.shape[3]
    n_planes = x_contig.shape[0] * x_contig.shape[1]
    n_hw_tiles = triton.cdiv(hw, _BLOCK_HW)
    total_tiles = n_planes * n_hw_tiles
    bias_contig = bias_1d.to(device=x.device, dtype=x.dtype).contiguous()

    if total_tiles > _MAX_PROGRAMS:
        _min_bias_scale_persistent_kernel[(_MAX_PROGRAMS, )](
            x_contig,
            bias_contig,
            out,
            total_tiles,
            hw,
            x_contig.shape[1],
            _MAX_PROGRAMS,
            float(constant_value),
            float(scaling_factor),
            BLOCK_HW=_BLOCK_HW,
            num_warps=8,
            num_stages=2,
        )
    else:
        _min_bias_scale_direct_kernel[(total_tiles, )](
            x_contig,
            bias_contig,
            out,
            total_tiles,
            hw,
            x_contig.shape[1],
            float(constant_value),
            float(scaling_factor),
            BLOCK_HW=_BLOCK_HW,
            num_warps=8,
            num_stages=2,
        )
    return out


class ModelNew(nn.Module):
    """Convolution followed by a fused min, bias add, and scaling epilogue."""

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        constant_value=DEFAULT_CONSTANT_VALUE,
        bias_shape=DEFAULT_BIAS_SHAPE,
        scaling_factor=DEFAULT_SCALING_FACTOR,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = float(constant_value)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        x = self.conv(x)
        return _apply_min_bias_scale(x, self.bias, self.constant_value,
                                     self.scaling_factor)


_MODEL_CACHE: dict[tuple[int | None, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError("run_operator expects input tensors on Ascend NPU")
    key = (x.device.index, x.dtype)
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
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="npu")]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, constant_value, bias_shape,
        scaling_factor
    ]
