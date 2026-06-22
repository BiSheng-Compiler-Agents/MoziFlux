import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK_W = 64


@triton.jit
def _maxpool3d_tile_kernel(
    x_ptr,
    y_ptr,
    total_tiles,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    outD: tl.constexpr,
    outH: tl.constexpr,
    outW: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_d: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    n_w_tiles = tl.cdiv(outW, BLOCK_W)
    w_tile = tile_id % n_w_tiles
    row = tile_id // n_w_tiles

    ow = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = ow < outW

    oh = row % outH
    t = row // outH
    od = t % outD
    t = t // outD
    c = t % C
    n = t // C

    in_z0 = od * stride_d - pad_d
    in_y0 = oh * stride_h - pad_h
    in_x0 = ow * stride_w - pad_w

    base_nc = (n * C + c) * D * H * W
    out_idx = (((n * C + c) * outD + od) * outH + oh) * outW + ow
    L1 = H * W
    L2 = W

    neg_inf = -float("inf")
    acc = tl.full((BLOCK_W, ), neg_inf, dtype=tl.float32)

    x = in_x0
    for kw in tl.static_range(0, K_W):
        x_valid = (x >= 0) & (x < W)
        for kd in tl.static_range(0, K_D):
            z = in_z0 + kd * dil_d
            z_valid = (z >= 0) & (z < D)
            z_base = z * L1
            for kh in tl.static_range(0, K_H):
                y = in_y0 + kh * dil_h
                y_valid = (y >= 0) & (y < H)
                base_zh = z_base + y * L2
                m = mask_ow & z_valid & y_valid & x_valid
                vals = tl.load(x_ptr + base_nc + base_zh + x,
                               mask=m,
                               other=neg_inf)
                acc = tl.maximum(acc, vals.to(tl.float32))
        x += dil_w

    tl.store(y_ptr + out_idx, acc, mask=mask_ow)


@triton.jit
def _maxpool3d_persistent_kernel(
    x_ptr,
    y_ptr,
    total_tiles,
    n_programs,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    outD: tl.constexpr,
    outH: tl.constexpr,
    outW: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_d: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    n_w_tiles = tl.cdiv(outW, BLOCK_W)
    for tile_id in range(pid, total_tiles, n_programs):
        w_tile = tile_id % n_w_tiles
        row = tile_id // n_w_tiles

        ow = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_ow = ow < outW

        oh = row % outH
        t = row // outH
        od = t % outD
        t = t // outD
        c = t % C
        n = t // C

        in_z0 = od * stride_d - pad_d
        in_y0 = oh * stride_h - pad_h
        in_x0 = ow * stride_w - pad_w

        base_nc = (n * C + c) * D * H * W
        out_idx = (((n * C + c) * outD + od) * outH + oh) * outW + ow
        L1 = H * W
        L2 = W

        neg_inf = -float("inf")
        acc = tl.full((BLOCK_W, ), neg_inf, dtype=tl.float32)

        for kd in tl.static_range(0, K_D):
            z = in_z0 + kd * dil_d
            z_valid = (z >= 0) & (z < D)
            z_base = z * L1
            for kh in tl.static_range(0, K_H):
                y = in_y0 + kh * dil_h
                y_valid = (y >= 0) & (y < H)
                base_zh = z_base + y * L2
                for kw in tl.static_range(0, K_W):
                    x = in_x0 + kw * dil_w
                    x_valid = (x >= 0) & (x < W)
                    m = mask_ow & z_valid & y_valid & x_valid
                    vals = tl.load(x_ptr + base_nc + base_zh + x,
                                   mask=m,
                                   other=neg_inf)
                    acc = tl.maximum(acc, vals.to(tl.float32))

        tl.store(y_ptr + out_idx, acc, mask=mask_ow)


def _as_triple(v):
    if isinstance(v, (tuple, list)):
        assert len(v) == 3
        return int(v[0]), int(v[1]), int(v[2])
    v = int(v)
    return (v, v, v)


def _compute_out_dim(in_size: int, k: int, stride: int, pad: int, dil: int,
                     ceil_mode: bool) -> int:
    eff = dil * (k - 1) + 1
    if ceil_mode:
        return max(0, (in_size + 2 * pad - eff + stride) // stride)
    return max(0, (in_size + 2 * pad - eff) // stride + 1)


class ModelNew(nn.Module):
    """Max Pooling 3D optimized for NCDHW contiguous tensors on Ascend NPU."""

    def __init__(self,
                 kernel_size: int,
                 stride: int = None,
                 padding: int = 0,
                 dilation: int = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.return_indices or self.ceil_mode or (
                not x.is_contiguous()) or x.dtype not in (
                    torch.float16, torch.float32) or x.device.type != "npu":
            return F.max_pool3d(x, self.kernel_size, self.stride, self.padding,
                                self.dilation, self.ceil_mode,
                                self.return_indices)

        N, C, D, H, W = x.shape
        kD, kH, kW = _as_triple(self.kernel_size)
        sD, sH, sW = _as_triple(self.stride)
        pD, pH, pW = _as_triple(self.padding)
        dD, dH, dW = _as_triple(self.dilation)

        outD = _compute_out_dim(D, kD, sD, pD, dD, self.ceil_mode)
        outH = _compute_out_dim(H, kH, sH, pH, dH, self.ceil_mode)
        outW = _compute_out_dim(W, kW, sW, pW, dW, self.ceil_mode)
        if outD == 0 or outH == 0 or outW == 0:
            return x.new_empty((N, C, outD, outH, outW))

        y = torch.empty((N, C, outD, outH, outW),
                        device=x.device,
                        dtype=x.dtype)
        total_tiles = N * C * outD * outH * triton.cdiv(outW, _BLOCK_W)
        common = dict(
            N=N,
            C=C,
            D=D,
            H=H,
            W=W,
            outD=outD,
            outH=outH,
            outW=outW,
            stride_d=sD,
            stride_h=sH,
            stride_w=sW,
            pad_d=pD,
            pad_h=pH,
            pad_w=pW,
            dil_d=dD,
            dil_h=dH,
            dil_w=dW,
            K_D=kD,
            K_H=kH,
            K_W=kW,
            BLOCK_W=_BLOCK_W,
        )
        if total_tiles <= _MAX_PROGRAMS:
            _maxpool3d_tile_kernel[(total_tiles, )](x, y, total_tiles,
                                                    **common)
        else:
            n_programs = _MAX_PROGRAMS
            _maxpool3d_persistent_kernel[(n_programs, )](x, y, total_tiles,
                                                         n_programs, **common)
        return y


def max_pool3d(
    x: torch.Tensor,
    kernel_size_: int | None = None,
    stride_: int | None = None,
    padding_: int | None = None,
    dilation_: int | None = None,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> torch.Tensor:
    if kernel_size_ is None:
        kernel_size_ = kernel_size
    if stride_ is None:
        stride_ = stride
    if padding_ is None:
        padding_ = padding
    if dilation_ is None:
        dilation_ = dilation
    return ModelNew(kernel_size_, stride_, padding_, dilation_, return_indices,
                    ceil_mode)(x)


batch_size = 16
channels = 32
dim1 = 128
dim2 = 128
dim3 = 128
kernel_size = 3
stride = 2
padding = 1
dilation = 3


def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
