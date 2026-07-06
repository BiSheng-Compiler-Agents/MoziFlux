import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 64
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 64
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_KERNEL_SIZE = 3
DEFAULT_MULTIPLIER_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1)
_MAX_GRID = 65535
_BLOCK_HW = 256


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _fused_scale_lrelu_gelu_nc_loop(
    x_ptr,
    m_ptr,
    y_ptr,
    NC,
    HW,
    C,
    negative_slope: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    c_idx = pid_nc % C
    scale = tl.load(m_ptr + c_idx).to(tl.float32)
    offs = tl.arange(0, BLOCK_HW)
    tl.multiple_of(offs, 16)
    base = pid_nc * HW
    for hw0 in tl.range(0, HW, BLOCK_HW, num_stages=2):
        hw = hw0 + offs
        mask = hw < HW
        x = tl.load(x_ptr + base + hw, mask=mask, other=0.0, care_padding=False).to(tl.float32)
        v = x * scale
        v = tl.where(v >= 0.0, v, v * negative_slope)
        y = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))
        tl.store(y_ptr + base + hw, y, mask=mask)


@triton.jit
def _fused_scale_lrelu_gelu_nc_persistent(
    x_ptr,
    m_ptr,
    y_ptr,
    NC,
    HW,
    C,
    n_programs,
    negative_slope: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HW)
    tl.multiple_of(offs, 16)
    for pid_nc in tl.range(pid, NC, n_programs, num_stages=2):
        c_idx = pid_nc % C
        scale = tl.load(m_ptr + c_idx).to(tl.float32)
        base = pid_nc * HW
        for hw0 in tl.range(0, HW, BLOCK_HW, num_stages=2):
            hw = hw0 + offs
            mask = hw < HW
            x = tl.load(x_ptr + base + hw, mask=mask, other=0.0, care_padding=False).to(tl.float32)
            v = x * scale
            v = tl.where(v >= 0.0, v, v * negative_slope)
            y = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))
            tl.store(y_ptr + base + hw, y, mask=mask)


class ModelNew(nn.Module):
    """
    Conv2d followed by y = GELU(LeakyReLU(conv(x) * per-channel multiplier)).
    The optimized Triton epilogue maps one program to each N*C plane and loops over
    contiguous HW tiles, avoiding per-element channel div/mod and large grid launches.
    """

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        multiplier_shape=DEFAULT_MULTIPLIER_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")

        x = self.conv(x).contiguous()
        N, C, H, W = x.shape
        if self.multiplier.numel() != C:
            raise RuntimeError(
                f"Multiplier must contain exactly one value per channel, got {self.multiplier.numel()} for C={C}"
            )
        out = torch.empty_like(x)
        m = self.multiplier.reshape(-1).to(device=x.device, dtype=x.dtype).contiguous()
        NC = N * C
        HW = H * W
        if NC <= _MAX_GRID:
            _fused_scale_lrelu_gelu_nc_loop[(NC,)](
                x, m, out, NC, HW, C, self.leaky_relu.negative_slope,
                BLOCK_HW=_BLOCK_HW, num_warps=4, num_stages=2,
            )
        else:
            n_programs = _MAX_GRID
            _fused_scale_lrelu_gelu_nc_persistent[(n_programs,)](
                x, m, out, NC, HW, C, n_programs, self.leaky_relu.negative_slope,
                BLOCK_HW=_BLOCK_HW, num_warps=4, num_stages=2,
            )
        return out


_MODEL_CACHE: dict[tuple[torch.device, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    return model(x)


batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='npu')]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
