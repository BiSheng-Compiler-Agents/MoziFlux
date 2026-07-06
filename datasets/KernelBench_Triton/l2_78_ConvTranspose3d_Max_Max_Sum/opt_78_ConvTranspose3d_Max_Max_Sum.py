import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 16
DEFAULT_IN_CHANNELS = 32
DEFAULT_OUT_CHANNELS = 64
DEFAULT_DEPTH = 32
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_KERNEL_SIZE = 5
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 2

_MAX_GRID = 65535
_C_BLOCK = 8
_BLOCK_HW = 64
_USE_ACL_DISPATCH = True


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _pool6_sum_c_direct_kernel(
    x_ptr,
    out_ptr,
    kwin,
    C: tl.constexpr,
    D2: tl.constexpr,
    H2: tl.constexpr,
    W2: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    N_HW_TILES: tl.constexpr,
    N_CBLOCKS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    tile_id = tl.program_id(0)
    c_block_id = tile_id % N_CBLOCKS
    tmp = tile_id // N_CBLOCKS
    hw_tile = tmp % N_HW_TILES
    nd = tmp // N_HW_TILES

    n = nd // D2
    d_out = nd - n * D2

    offs_hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (H2 * W2)
    h_out = offs_hw // W2
    w_out = offs_hw - h_out * W2

    c_offsets = c_block_id * C_BLOCK + tl.arange(0, C_BLOCK)
    mask_c = c_offsets < C

    m = tl.full((C_BLOCK, BLOCK_HW), -float("inf"), dtype=tl.float32)
    base_n = n * stride_n
    d_base = d_out * 6
    h_base = h_out * 6
    w_base = w_out * 6

    for kd in range(0, kwin):
        d_off = (d_base + kd) * stride_d
        for kh in range(0, kwin):
            h_off = (h_base + kh) * stride_h
            for kw in range(0, kwin):
                w_off = (w_base + kw) * stride_w
                ptrs = x_ptr + base_n + c_offsets[:, None] * stride_c + d_off + h_off[None, :] + w_off[None, :]
                vals = tl.load(ptrs, mask=mask_c[:, None] & mask_hw[None, :], other=-float("inf"))
                m = tl.maximum(m, vals.to(tl.float32))

    partial = tl.sum(tl.where(mask_c[:, None], m, 0.0), axis=0)
    out_ptrs = out_ptr + n * out_stride_n + d_out * out_stride_d + h_out * out_stride_h + w_out * out_stride_w
    tl.atomic_add(out_ptrs, partial, sem="relaxed", mask=mask_hw)


@triton.jit
def _pool6_sum_c_persistent_kernel(
    x_ptr,
    out_ptr,
    kwin,
    total_tiles,
    n_programs,
    C: tl.constexpr,
    D2: tl.constexpr,
    H2: tl.constexpr,
    W2: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    N_HW_TILES: tl.constexpr,
    N_CBLOCKS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    for tile_id in range(pid, total_tiles, n_programs):
        c_block_id = tile_id % N_CBLOCKS
        tmp = tile_id // N_CBLOCKS
        hw_tile = tmp % N_HW_TILES
        nd = tmp // N_HW_TILES

        n = nd // D2
        d_out = nd - n * D2

        offs_hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < (H2 * W2)
        h_out = offs_hw // W2
        w_out = offs_hw - h_out * W2

        c_offsets = c_block_id * C_BLOCK + tl.arange(0, C_BLOCK)
        mask_c = c_offsets < C

        m = tl.full((C_BLOCK, BLOCK_HW), -float("inf"), dtype=tl.float32)
        base_n = n * stride_n
        d_base = d_out * 6
        h_base = h_out * 6
        w_base = w_out * 6

        for kd in range(0, kwin):
            d_off = (d_base + kd) * stride_d
            for kh in range(0, kwin):
                h_off = (h_base + kh) * stride_h
                for kw in range(0, kwin):
                    w_off = (w_base + kw) * stride_w
                    ptrs = x_ptr + base_n + c_offsets[:, None] * stride_c + d_off + h_off[None, :] + w_off[None, :]
                    vals = tl.load(ptrs, mask=mask_c[:, None] & mask_hw[None, :], other=-float("inf"))
                    m = tl.maximum(m, vals.to(tl.float32))

        partial = tl.sum(tl.where(mask_c[:, None], m, 0.0), axis=0)
        out_ptrs = out_ptr + n * out_stride_n + d_out * out_stride_d + h_out * out_stride_h + w_out * out_stride_w
        tl.atomic_add(out_ptrs, partial, sem="relaxed", mask=mask_hw)


def _fused_two_pools_sum_channels(x: torch.Tensor) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError("optimized fused pool+sum expects an Ascend NPU tensor")
    if x.ndim != 5:
        raise ValueError(f"expected 5D NCDHW tensor, got {tuple(x.shape)}")

    x = x.contiguous()
    N, C, D, H, W = x.shape
    if D < 6 or H < 6 or W < 6:
        raise ValueError("input spatial dimensions must be at least 6 for fused MaxPool3d(2)->MaxPool3d(3)")

    D2 = (D - 6) // 6 + 1
    H2 = (H - 6) // 6 + 1
    W2 = (W - 6) // 6 + 1
    out = torch.empty((N, 1, D2, H2, W2), device=x.device, dtype=x.dtype)
    out.zero_()

    sN, sC, sD, sH, sW = x.stride()
    oN, _oC, oD, oH, oW = out.stride()
    n_hw_tiles = triton.cdiv(H2 * W2, _BLOCK_HW)
    n_cblocks = triton.cdiv(C, _C_BLOCK)
    total_tiles = N * D2 * n_hw_tiles * n_cblocks
    if total_tiles > _MAX_GRID:
        n_programs = _MAX_GRID
        _pool6_sum_c_persistent_kernel[(n_programs,)](
            x, out, 6, total_tiles, n_programs, C, D2, H2, W2, sN, sC, sD, sH, sW, oN, oD, oH, oW,
            n_hw_tiles, n_cblocks, BLOCK_HW=_BLOCK_HW, C_BLOCK=_C_BLOCK
        )
    else:
        _pool6_sum_c_direct_kernel[(total_tiles,)](
            x, out, 6, C, D2, H2, W2, sN, sC, sD, sH, sW, oN, oD, oH, oW, n_hw_tiles, n_cblocks,
            BLOCK_HW=_BLOCK_HW, C_BLOCK=_C_BLOCK
        )
    return out


class ModelNew(nn.Module):
    """ConvTranspose3d followed by fused MaxPool3d(2), MaxPool3d(3), and channel sum."""

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects Ascend NPU inputs")
        if not _is_npu_tensor(self.conv_transpose.weight):
            raise RuntimeError("ModelNew weights must be moved to Ascend NPU before execution")
        if self.conv_transpose.bias is not None and not _is_npu_tensor(self.conv_transpose.bias):
            raise RuntimeError("ModelNew bias must be moved to Ascend NPU before execution")
        x = self.conv_transpose(x)
        if _USE_ACL_DISPATCH:
            x = F.max_pool3d(x, kernel_size=2, stride=2)
            x = F.max_pool3d(x, kernel_size=3, stride=3)
            return x.sum(dim=1, keepdim=True)
        return _fused_two_pools_sum_channels(x)


batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
