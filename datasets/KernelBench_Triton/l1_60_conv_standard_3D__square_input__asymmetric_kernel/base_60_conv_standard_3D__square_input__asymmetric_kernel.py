import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


DEFAULT_IN_CHANNELS = 3
DEFAULT_OUT_CHANNELS = 64
DEFAULT_KERNEL_SIZE = (3, 5, 7)
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 0
DEFAULT_DILATION = 1
DEFAULT_GROUPS = 1
DEFAULT_BIAS = False

KERNEL_D = DEFAULT_KERNEL_SIZE[0]
KERNEL_H = DEFAULT_KERNEL_SIZE[1]
KERNEL_W = DEFAULT_KERNEL_SIZE[2]
KERNEL_VOLUME = KERNEL_D * KERNEL_H * KERNEL_W


@triton.jit
def _conv3d_implicit_gemm_kernel(
    x_ptr,
    w_ptr,
    x_delta_ptr,
    out_ptr,
    total_positions,
    in_d_size,
    in_h_size,
    in_w_size,
    out_d_size,
    out_h_size,
    out_w_size,
    out_hw,
    out_dhw,
    batch_out_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KERNEL_VOL: tl.constexpr,
    KERNEL_H_CONST: tl.constexpr,
    KERNEL_W_CONST: tl.constexpr,
    IN_C: tl.constexpr,
    OUT_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < total_positions
    n_mask = offs_n < OUT_C

    batch_idx = offs_m // out_dhw
    spatial_idx = offs_m % out_dhw
    out_d_idx = spatial_idx // out_hw
    rem_hw = spatial_idx % out_hw
    out_h_idx = rem_hw // out_w_size
    out_w_idx = rem_hw % out_w_size
    batch_base = batch_idx * (IN_C * in_d_size * in_h_size * in_w_size)
    spatial_base = out_d_idx * (in_h_size * in_w_size) + out_h_idx * in_w_size + out_w_idx
    x_mask_base = m_mask[:, None]
    w_mask_base = n_mask[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_base in range(0, KERNEL_VOL * IN_C, BLOCK_K):
        offs_k = k_base + tl.arange(0, BLOCK_K)
        k_mask = offs_k < (KERNEL_VOL * IN_C)
        x_delta = tl.load(x_delta_ptr + offs_k, mask=k_mask, other=0)

        x_offsets = (
            batch_base[:, None]
            + spatial_base[:, None]
            + x_delta[None, :]
        )
        x_mask = x_mask_base & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        tl.compile_hint(x_tile, "dot_pad_only_k")

        w_offsets = offs_k[:, None] * OUT_C + offs_n[None, :]
        w_mask = k_mask[:, None] & w_mask_base
        w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)
        tl.compile_hint(w_tile, "dot_pad_only_k")

        acc = tl.dot(x_tile, w_tile, acc)

    out_offsets = (
        batch_idx[:, None] * batch_out_stride
        + offs_n[None, :] * out_dhw
        + out_d_idx[:, None] * out_hw
        + out_h_idx[:, None] * out_w_size
        + out_w_idx[:, None]
    )
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offsets, acc.to(tl.float16), mask=out_mask)


def _pack_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.permute(1, 2, 3, 4, 0).contiguous().view(KERNEL_VOLUME * DEFAULT_IN_CHANNELS, DEFAULT_OUT_CHANNELS)


def _build_x_delta_table(in_d_size: int, in_h_size: int, in_w_size: int, device: torch.device) -> torch.Tensor:
    kernel_idx = torch.arange(KERNEL_VOLUME * DEFAULT_IN_CHANNELS, device=device, dtype=torch.int32)
    c_idx = torch.div(kernel_idx, KERNEL_VOLUME, rounding_mode="floor")
    kernel_only = torch.remainder(kernel_idx, KERNEL_VOLUME)
    kd_idx = torch.div(kernel_only, KERNEL_H * KERNEL_W, rounding_mode="floor")
    kh_kw_idx = torch.remainder(kernel_only, KERNEL_H * KERNEL_W)
    kh_idx = torch.div(kh_kw_idx, KERNEL_W, rounding_mode="floor")
    kw_idx = torch.remainder(kh_kw_idx, KERNEL_W)
    channel_stride = in_d_size * in_h_size * in_w_size
    depth_stride = in_h_size * in_w_size
    return (c_idx * channel_stride + kd_idx * depth_stride + kh_idx * in_w_size + kw_idx).to(torch.int32)


def _can_use_triton_path(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.device.type == "npu"
        and x.dtype == torch.float16
        and x.is_contiguous()
        and weight.dtype == torch.float16
        and weight.is_contiguous()
        and x.ndim == 5
        and tuple(weight.shape) == (DEFAULT_OUT_CHANNELS, DEFAULT_IN_CHANNELS, KERNEL_D, KERNEL_H, KERNEL_W)
        and x.shape[1] == DEFAULT_IN_CHANNELS
        and x.shape[2] >= KERNEL_D
        and x.shape[3] >= KERNEL_H
        and x.shape[4] >= KERNEL_W
    )


def _conv3d_triton_forward(x: torch.Tensor, packed_weight: torch.Tensor, x_delta: torch.Tensor) -> torch.Tensor:
    batch_size, _, in_d_size, in_h_size, in_w_size = x.shape
    out_d_size = in_d_size - KERNEL_D + 1
    out_h_size = in_h_size - KERNEL_H + 1
    out_w_size = in_w_size - KERNEL_W + 1
    out = torch.empty(
        (batch_size, DEFAULT_OUT_CHANNELS, out_d_size, out_h_size, out_w_size),
        device=x.device,
        dtype=x.dtype,
    )

    total_positions = batch_size * out_d_size * out_h_size * out_w_size
    out_hw = out_h_size * out_w_size
    out_dhw = out_d_size * out_hw
    batch_out_stride = DEFAULT_OUT_CHANNELS * out_dhw
    grid = (
        triton.cdiv(total_positions, 224),
        triton.cdiv(DEFAULT_OUT_CHANNELS, 64),
    )
    _conv3d_implicit_gemm_kernel[grid](
        x,
        packed_weight,
        x_delta,
        out,
        total_positions,
        in_d_size,
        in_h_size,
        in_w_size,
        out_d_size,
        out_h_size,
        out_w_size,
        out_hw,
        out_dhw,
        batch_out_stride,
        BLOCK_M=224,
        BLOCK_N=64,
        BLOCK_K=32,
        KERNEL_VOL=KERNEL_VOLUME,
        KERNEL_H_CONST=KERNEL_H,
        KERNEL_W_CONST=KERNEL_W,
        IN_C=DEFAULT_IN_CHANNELS,
        OUT_C=DEFAULT_OUT_CHANNELS,
    )
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_height, kernel_width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: tuple = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        dilation: int = DEFAULT_DILATION,
        groups: int = DEFAULT_GROUPS,
        bias: bool = DEFAULT_BIAS,
    ):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ).to("npu")
        self._packed_weight = None
        self._packed_weight_key = None
        self._x_delta = None
        self._x_delta_key = None

    def _get_packed_weight(self) -> torch.Tensor:
        weight = self.conv3d.weight
        packed_weight_key = (weight.data_ptr(), weight.dtype)
        if self._packed_weight_key != packed_weight_key:
            self._packed_weight = _pack_weight(weight)
            self._packed_weight_key = packed_weight_key
        return self._packed_weight

    def _get_x_delta(self, x: torch.Tensor) -> torch.Tensor:
        x_delta_key = (x.device, tuple(x.shape[2:]))
        if self._x_delta_key != x_delta_key:
            in_d_size, in_h_size, in_w_size = x.shape[2:]
            self._x_delta = _build_x_delta_table(in_d_size, in_h_size, in_w_size, x.device)
            self._x_delta_key = x_delta_key
        return self._x_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects an Ascend NPU tensor input")

        if self.conv3d.weight.device != x.device or self.conv3d.weight.dtype != x.dtype:
            self.conv3d = self.conv3d.to(device=x.device, dtype=x.dtype)
            self._packed_weight = None
            self._packed_weight_key = None
            self._x_delta = None
            self._x_delta_key = None

        if (
            self.conv3d.bias is None
            and self.conv3d.groups == 1
            and self.conv3d.stride == (1, 1, 1)
            and self.conv3d.padding == (0, 0, 0)
            and self.conv3d.dilation == (1, 1, 1)
            and tuple(self.conv3d.weight.shape) == (DEFAULT_OUT_CHANNELS, DEFAULT_IN_CHANNELS, KERNEL_D, KERNEL_H, KERNEL_W)
            and _can_use_triton_path(x, self.conv3d.weight)
        ):
            return _conv3d_triton_forward(x, self._get_packed_weight(), self._get_x_delta(x))

        return F.conv3d(
            x,
            self.conv3d.weight,
            self.conv3d.bias,
            stride=self.conv3d.stride,
            padding=self.conv3d.padding,
            dilation=self.conv3d.dilation,
            groups=self.conv3d.groups,
        )


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
width = 64
height = 64
depth = 64


def get_inputs():
    x = torch.rand(batch_size, in_channels, width, height, depth)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
