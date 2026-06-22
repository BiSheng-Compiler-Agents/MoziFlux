import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK_W = 32
_ACL_TILE_THRESHOLD = 1024


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


@triton.jit
def _fused_minD_softmaxC_direct(
    x_ptr,
    y_ptr,
    B,
    C,
    D,
    H,
    W,
    stride_n,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    TOT_D_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_w_tiles = tl.cdiv(W, BLOCK_W)
    w_tile = pid % num_w_tiles
    pid = pid // num_w_tiles
    h_idx = pid % H
    n_idx = pid // H

    c_offsets = tl.arange(0, BLOCK_C)
    w_offsets = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
    c_mask = c_offsets < C
    w_mask = w_offsets < W

    run_min = tl.full([BLOCK_C, BLOCK_W], float("inf"), dtype=tl.float32)
    in_base = n_idx * stride_n + h_idx * stride_h
    for dt in tl.static_range(0, TOT_D_TILES):
        d_offsets = dt * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        ptrs = (x_ptr + in_base + c_offsets[:, None, None] * stride_c +
                d_offsets[None, :, None] * stride_d +
                w_offsets[None, None, :] * stride_w)
        mask = c_mask[:, None, None] & d_mask[None, :, None] & w_mask[None,
                                                                      None, :]
        vals = tl.load(ptrs, mask=mask, other=float("inf"),
                       care_padding=False).to(tl.float32)
        run_min = tl.minimum(run_min, tl.min(vals, axis=1))

    valid_min = tl.where(c_mask[:, None], run_min, -float("inf"))
    x_max = tl.max(valid_min, axis=0)
    exps = tl.exp(run_min - x_max[None, :])
    exps = tl.where(c_mask[:, None] & w_mask[None, :], exps, 0.0)
    denom = tl.sum(exps, axis=0)
    out_vals = exps / denom[None, :]

    out_ptrs = (y_ptr + n_idx * out_stride_n +
                c_offsets[:, None] * out_stride_c + h_idx * out_stride_h +
                w_offsets[None, :] * out_stride_w)
    tl.store(out_ptrs, out_vals, mask=c_mask[:, None] & w_mask[None, :])


@triton.jit
def _fused_minD_softmaxC_persistent(
    x_ptr,
    y_ptr,
    B,
    C,
    D,
    H,
    W,
    stride_n,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    total_tiles,
    n_programs,
    TOT_D_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    tile = tl.program_id(axis=0)
    num_w_tiles = tl.cdiv(W, BLOCK_W)
    while tile < total_tiles:
        w_tile = tile % num_w_tiles
        tmp = tile // num_w_tiles
        h_idx = tmp % H
        n_idx = tmp // H

        c_offsets = tl.arange(0, BLOCK_C)
        w_offsets = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
        c_mask = c_offsets < C
        w_mask = w_offsets < W

        run_min = tl.full([BLOCK_C, BLOCK_W], float("inf"), dtype=tl.float32)
        in_base = n_idx * stride_n + h_idx * stride_h
        for dt in tl.static_range(0, TOT_D_TILES):
            d_offsets = dt * BLOCK_D + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < D
            ptrs = (x_ptr + in_base + c_offsets[:, None, None] * stride_c +
                    d_offsets[None, :, None] * stride_d +
                    w_offsets[None, None, :] * stride_w)
            mask = c_mask[:, None, None] & d_mask[None, :,
                                                  None] & w_mask[None, None, :]
            vals = tl.load(ptrs,
                           mask=mask,
                           other=float("inf"),
                           care_padding=False).to(tl.float32)
            run_min = tl.minimum(run_min, tl.min(vals, axis=1))

        valid_min = tl.where(c_mask[:, None], run_min, -float("inf"))
        x_max = tl.max(valid_min, axis=0)
        exps = tl.exp(run_min - x_max[None, :])
        exps = tl.where(c_mask[:, None] & w_mask[None, :], exps, 0.0)
        denom = tl.sum(exps, axis=0)
        out_vals = exps / denom[None, :]

        out_ptrs = (y_ptr + n_idx * out_stride_n +
                    c_offsets[:, None] * out_stride_c + h_idx * out_stride_h +
                    w_offsets[None, :] * out_stride_w)
        tl.store(out_ptrs, out_vals, mask=c_mask[:, None] & w_mask[None, :])
        tile += n_programs


def _fused_triton(x: torch.Tensor) -> torch.Tensor:
    B, C, D, H, W = x.shape
    y = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
    sN, sC, sD, sH, sW = x.stride()
    oN, oC, oH, oW = y.stride()
    block_c = _next_pow2(C)
    block_d = min(_next_pow2(D), 64)
    tot_d_tiles = triton.cdiv(D, block_d)
    num_w_tiles = triton.cdiv(W, _BLOCK_W)
    total_tiles = B * H * num_w_tiles

    if total_tiles > _MAX_PROGRAMS:
        _fused_minD_softmaxC_persistent[(_MAX_PROGRAMS, )](
            x,
            y,
            B,
            C,
            D,
            H,
            W,
            sN,
            sC,
            sD,
            sH,
            sW,
            oN,
            oC,
            oH,
            oW,
            total_tiles,
            _MAX_PROGRAMS,
            TOT_D_TILES=tot_d_tiles,
            BLOCK_C=block_c,
            BLOCK_D=block_d,
            BLOCK_W=_BLOCK_W,
            num_warps=4,
            num_stages=2,
        )
    else:
        _fused_minD_softmaxC_direct[(total_tiles, )](
            x,
            y,
            B,
            C,
            D,
            H,
            W,
            sN,
            sC,
            sD,
            sH,
            sW,
            oN,
            oC,
            oH,
            oW,
            TOT_D_TILES=tot_d_tiles,
            BLOCK_C=block_c,
            BLOCK_D=block_d,
            BLOCK_W=_BLOCK_W,
            num_warps=4,
            num_stages=2,
        )
    return y


class ModelNew(nn.Module):
    """Conv3d -> min over depth (dim=2 after conv) -> softmax over channels."""

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 16,
                 kernel_size: int = 3,
                 dim: int = 2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew requires Ascend NPU input tensors.")
        if self.dim != 2:
            raise NotImplementedError(
                "ModelNew only supports reduction along dim=2.")
        x = self.conv(x).contiguous()
        _, C, _, H, W = x.shape
        total_tiles = x.shape[0] * H * triton.cdiv(W, _BLOCK_W)
        if _next_pow2(C) <= 64 and total_tiles < _ACL_TILE_THRESHOLD:
            return _fused_triton(x)
        return F.softmax(torch.amin(x, dim=2), dim=1)


batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]
