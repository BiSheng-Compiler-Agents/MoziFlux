import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 8
DEFAULT_OUT_CHANNELS = 32
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_KERNEL_SIZE = 3
DEFAULT_NUM_GROUPS = 8
_MAX_PROGRAMS = 65535
_BLOCK_HW = 4096


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _bias_scale_sigmoid_row_direct(
    x_ptr,
    bias_ptr,
    scale_ptr,
    y_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    N_HW_TILES: tl.constexpr,
    TOTAL_TILES,
    BLOCK_HW: tl.constexpr,
):
    tile_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HW)
    row = tile_id // N_HW_TILES
    hw_tile = tile_id - row * N_HW_TILES
    hw_offs = hw_tile * BLOCK_HW + offs
    mask = (tile_id < TOTAL_TILES) & (hw_offs < HW)
    c_idx = row % C
    ptrs = x_ptr + row * HW + hw_offs

    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + c_idx).to(tl.float32)
    s = tl.load(scale_ptr + c_idx).to(tl.float32)
    z = (x + b) * s
    out = 1.0 / (1.0 + tl.exp(-z))
    tl.store(y_ptr + row * HW + hw_offs, out, mask=mask)


@triton.jit
def _bias_scale_sigmoid_row_persistent(
    x_ptr,
    bias_ptr,
    scale_ptr,
    y_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    N_HW_TILES: tl.constexpr,
    TOTAL_TILES,
    N_PROGRAMS,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HW)
    for tile_id in range(pid, TOTAL_TILES, N_PROGRAMS):
        row = tile_id // N_HW_TILES
        hw_tile = tile_id - row * N_HW_TILES
        hw_offs = (hw_tile * BLOCK_HW + offs).to(tl.int64)
        mask = hw_offs < HW
        c_idx = row % C
        base = (row * HW).to(tl.int64)
        ptrs = x_ptr + base + hw_offs

        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + c_idx).to(tl.float32)
        s = tl.load(scale_ptr + c_idx).to(tl.float32)
        z = (x + b) * s
        out = 1.0 / (1.0 + tl.exp(-z))
        tl.store(y_ptr + base + hw_offs, out, mask=mask)


def fused_bias_scale_sigmoid(x: torch.Tensor, bias: torch.Tensor,
                             scale: torch.Tensor):
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "fused_bias_scale_sigmoid expects input tensors on Ascend NPU")
    if x.requires_grad:
        raise RuntimeError(
            "fused_bias_scale_sigmoid does not support autograd-enabled inputs"
        )
    if x.dtype not in (torch.float16, torch.float32):
        raise RuntimeError(
            "fused_bias_scale_sigmoid supports only float16 and float32 inputs"
        )

    x_contig = x.contiguous()
    bias_contig = bias.contiguous()
    scale_contig = scale.contiguous()
    N, C, H, W = x_contig.shape
    HW = H * W
    n_hw_tiles = triton.cdiv(HW, _BLOCK_HW)
    total_tiles = N * C * n_hw_tiles
    y = torch.empty_like(x_contig)

    if total_tiles > _MAX_PROGRAMS:
        n_programs = _MAX_PROGRAMS
        _bias_scale_sigmoid_row_persistent[(n_programs, )](
            x_contig,
            bias_contig,
            scale_contig,
            y,
            HW,
            C,
            n_hw_tiles,
            total_tiles,
            n_programs,
            BLOCK_HW=_BLOCK_HW,
        )
    else:
        _bias_scale_sigmoid_row_direct[(total_tiles, )](
            x_contig,
            bias_contig,
            scale_contig,
            y,
            HW,
            C,
            n_hw_tiles,
            total_tiles,
            BLOCK_HW=_BLOCK_HW,
        )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        num_groups=DEFAULT_NUM_GROUPS,
        bias_shape=None,
        scale_shape=None,
    ):
        super(ModelNew, self).__init__()
        if bias_shape is None:
            bias_shape = (out_channels, 1, 1)
        if scale_shape is None:
            scale_shape = (out_channels, 1, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")
        if x.dtype not in (torch.float16, torch.float32):
            raise RuntimeError(
                "ModelNew supports only float16 and float32 inputs")

        x = self.conv(x)
        x = fused_bias_scale_sigmoid(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x


_MODEL_CACHE: dict[tuple[str, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (str(x.device), x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    return model(x)


batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='npu')]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, num_groups, bias_shape,
        scale_shape
    ]
