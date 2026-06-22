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
DEFAULT_OUT_CHANNELS = 64
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 1
DEFAULT_OUTPUT_PADDING = 1
DEFAULT_ADD_VALUE = 0.5
DEFAULT_SCALE = 2

_MAX_PROGRAMS = 65535
_BLOCK_SIZE = 4096


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _mish_add_hardtanh_scale_direct_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    add_value,
    scale_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last")
    x_f32 = x.to(tl.float32)
    ax = tl.abs(x_f32)
    sp = tl.maximum(x_f32, 0.0) + tl.log(1.0 + tl.exp(-ax))
    tanh_sp = 2.0 / (1.0 + tl.exp(-(2.0 * sp))) - 1.0
    out = x_f32 * tanh_sp
    out = out + add_value
    out = tl.minimum(tl.maximum(out, -1.0), 1.0)
    out = (out * scale_value).to(x.dtype)
    tl.store(y_ptr + offs, out, mask=mask)


@triton.jit
def _mish_add_hardtanh_scale_persistent_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    n_programs,
    add_value,
    scale_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offs = (tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        tl.max_contiguous(offs, BLOCK_SIZE)
        tl.multiple_of(offs, 16)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_last")
        x_f32 = x.to(tl.float32)
        ax = tl.abs(x_f32)
        sp = tl.maximum(x_f32, 0.0) + tl.log(1.0 + tl.exp(-ax))
        tanh_sp = 2.0 / (1.0 + tl.exp(-(2.0 * sp))) - 1.0
        out = x_f32 * tanh_sp
        out = out + add_value
        out = tl.minimum(tl.maximum(out, -1.0), 1.0)
        out = (out * scale_value).to(x.dtype)
        tl.store(y_ptr + offs, out, mask=mask)


def _fused_mish_add_hardtanh_scale(x: torch.Tensor, add_value: float,
                                   scale: float) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "_fused_mish_add_hardtanh_scale expects input tensors on Ascend NPU"
        )
    if x.numel() == 0:
        return x.clone()

    x_contig = x.contiguous()
    n_elements = x_contig.numel()
    n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
    if n_tiles > _MAX_PROGRAMS:
        n_programs = _MAX_PROGRAMS
        _mish_add_hardtanh_scale_persistent_kernel[(n_programs, )](
            x_contig,
            x_contig,
            n_elements,
            n_programs,
            float(add_value),
            float(scale),
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
    else:
        _mish_add_hardtanh_scale_direct_kernel[(n_tiles, )](
            x_contig,
            x_contig,
            n_elements,
            float(add_value),
            float(scale),
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
    return x_contig


class ModelNew(nn.Module):
    """ConvTranspose2d followed by fused Mish + add + Hardtanh + scale."""

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
        output_padding=DEFAULT_OUTPUT_PADDING,
        add_value=DEFAULT_ADD_VALUE,
        scale=DEFAULT_SCALE,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
                                                 kernel_size, stride, padding,
                                                 output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        x = self.conv_transpose(x)
        return _fused_mish_add_hardtanh_scale(x, self.add_value, self.scale)


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
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='npu')]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, stride, padding,
        output_padding, add_value, scale
    ]
