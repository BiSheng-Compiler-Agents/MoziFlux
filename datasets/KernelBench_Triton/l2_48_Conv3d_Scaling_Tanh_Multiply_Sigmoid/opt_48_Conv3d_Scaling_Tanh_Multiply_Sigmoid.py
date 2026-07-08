import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 3
DEFAULT_OUT_CHANNELS = 16
DEFAULT_DEPTH = 16
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_SCALING_FACTOR = 2
DEFAULT_BIAS_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1, 1)

_MAX_GRID = 65535
_BLOCK_SIZE = 4096


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


@triton.jit
def _fused_pointwise_ncdhw_direct_kernel(
    x_ptr,  # *f32 contiguous NCDHW conv output, in-place
    sf_ptr,  # *f32, shape [C]
    bias_ptr,  # *f32, shape [C]
    out_ptr,  # *f32
    total_segments: tl.constexpr,  # N*C
    DHW: tl.constexpr,  # D*H*W
    tiles_per_segment: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_SIZE)

    seg = pid // tiles_per_segment
    tile_in_seg = pid - seg * tiles_per_segment
    hw = tile_in_seg * BLOCK_SIZE + offs
    mask = hw < DHW
    c = seg % 16
    base = seg * DHW + hw

    x = tl.load(x_ptr + base, mask=mask, other=0.0, care_padding=False)
    sf = tl.load(sf_ptr + c)
    b = tl.load(bias_ptr + c)
    x = x * sf
    sig2x = 1.0 / (1.0 + tl.exp(-2.0 * x))
    x = 2.0 * sig2x - 1.0
    x = x * b
    x = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + base, x, mask=mask)


@triton.jit
def _fused_pointwise_ncdhw_persistent_kernel(
    x_ptr,
    sf_ptr,
    bias_ptr,
    out_ptr,
    total_segments: tl.constexpr,
    DHW: tl.constexpr,
    tiles_per_segment: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    nprog = tl.num_programs(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_SIZE)

    total_tiles = total_segments * tiles_per_segment
    for tile_id in tl.range(pid, total_tiles, nprog, num_stages=2):
        seg = tile_id // tiles_per_segment
        tile_in_seg = tile_id - seg * tiles_per_segment
        hw = tile_in_seg * BLOCK_SIZE + offs
        mask = hw < DHW
        c = seg % 16
        base = seg * DHW + hw

        x = tl.load(x_ptr + base, mask=mask, other=0.0, care_padding=False)
        sf = tl.load(sf_ptr + c)
        b = tl.load(bias_ptr + c)
        x = x * sf
        sig2x = 1.0 / (1.0 + tl.exp(-2.0 * x))
        x = 2.0 * sig2x - 1.0
        x = x * b
        x = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + base, x, mask=mask)


@triton.jit
def _fused_pointwise_ncdhw_direct_generic_kernel(
    x_ptr,
    sf_ptr,
    bias_ptr,
    out_ptr,
    C: tl.constexpr,
    total_segments: tl.constexpr,
    DHW: tl.constexpr,
    tiles_per_segment: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_SIZE)

    seg = pid // tiles_per_segment
    tile_in_seg = pid - seg * tiles_per_segment
    hw = tile_in_seg * BLOCK_SIZE + offs
    mask = hw < DHW
    c = seg % C
    base = seg * DHW + hw

    x = tl.load(x_ptr + base, mask=mask, other=0.0, care_padding=False)
    sf = tl.load(sf_ptr + c)
    b = tl.load(bias_ptr + c)
    x = x * sf
    sig2x = 1.0 / (1.0 + tl.exp(-2.0 * x))
    x = 2.0 * sig2x - 1.0
    x = x * b
    x = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + base, x, mask=mask)


@triton.jit
def _fused_pointwise_ncdhw_persistent_generic_kernel(
    x_ptr,
    sf_ptr,
    bias_ptr,
    out_ptr,
    C: tl.constexpr,
    total_segments: tl.constexpr,
    DHW: tl.constexpr,
    tiles_per_segment: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    nprog = tl.num_programs(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_SIZE)

    total_tiles = total_segments * tiles_per_segment
    for tile_id in tl.range(pid, total_tiles, nprog, num_stages=2):
        seg = tile_id // tiles_per_segment
        tile_in_seg = tile_id - seg * tiles_per_segment
        hw = tile_in_seg * BLOCK_SIZE + offs
        mask = hw < DHW
        c = seg % C
        base = seg * DHW + hw

        x = tl.load(x_ptr + base, mask=mask, other=0.0, care_padding=False)
        sf = tl.load(sf_ptr + c)
        b = tl.load(bias_ptr + c)
        x = x * sf
        sig2x = 1.0 / (1.0 + tl.exp(-2.0 * x))
        x = 2.0 * sig2x - 1.0
        x = x * b
        x = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + base, x, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh,
    multiplies by the learned per-channel bias tensor, and applies sigmoid.
    """

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        scaling_factor=DEFAULT_SCALING_FACTOR,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor_value = scaling_factor
        self.scaling_factor = nn.Parameter(
            torch.full(bias_shape, float(scaling_factor)))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in {torch.float32, torch.bfloat16}:
            raise RuntimeError(
                f"ModelNew supports only float32 and bfloat16 inputs, got {x.dtype}"
            )

        x = self.conv(x).contiguous()
        _, C, D, H, W = x.shape
        DHW = D * H * W
        total_segments = x.numel() // DHW
        tiles_per_segment = triton.cdiv(DHW, _BLOCK_SIZE)
        total_tiles = total_segments * tiles_per_segment
        grid = (min(total_tiles, _MAX_GRID), )

        sf = self.scaling_factor.reshape(C).contiguous()
        bs = self.bias.reshape(C).contiguous()

        use_persistent = total_tiles > _MAX_GRID

        if C == 16 and not use_persistent:
            _fused_pointwise_ncdhw_direct_kernel[grid](
                x,
                sf,
                bs,
                x,
                total_segments,
                DHW,
                tiles_per_segment,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=8,
            )
        elif C == 16:
            _fused_pointwise_ncdhw_persistent_kernel[grid](
                x,
                sf,
                bs,
                x,
                total_segments,
                DHW,
                tiles_per_segment,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=8,
            )
        elif not use_persistent:
            _fused_pointwise_ncdhw_direct_generic_kernel[grid](
                x,
                sf,
                bs,
                x,
                C,
                total_segments,
                DHW,
                tiles_per_segment,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=8,
            )
        else:
            _fused_pointwise_ncdhw_persistent_generic_kernel[grid](
                x,
                sf,
                bs,
                x,
                C,
                total_segments,
                DHW,
                tiles_per_segment,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=8,
            )
        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]
