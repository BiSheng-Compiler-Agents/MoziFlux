import os

os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import torch_npu  # noqa: F401

_MAX_GRID = 65535


@triton.jit
def _maxpool2d_flat_kernel(
    x_ptr,
    y_ptr,
    total_out,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    n_tiles = tl.cdiv(total_out, BLOCK)

    for tile_id in range(pid, n_tiles, n_programs):
        offs = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total_out
        tl.max_contiguous(tl.arange(0, BLOCK), BLOCK)

        wo = offs % W_out
        tmp = offs // W_out
        ho = tmp % H_out
        tmp = tmp // H_out
        c = tmp % C
        n = tmp // C

        h_start = ho * STRIDE_H - PAD_H
        w_start = wo * STRIDE_W - PAD_W
        base_x = (n * C + c) * H * W
        max_val = tl.full((BLOCK, ), -float("inf"), tl.float32)

        for kh in tl.static_range(0, K_H):
            ih = h_start + kh * DIL_H
            ih_in = (ih >= 0) & (ih < H)
            safe_ih = tl.where(ih_in, ih, 0)
            row_base = base_x + safe_ih * W
            for kw in tl.static_range(0, K_W):
                iw = w_start + kw * DIL_W
                iw_in = (iw >= 0) & (iw < W)
                safe_iw = tl.where(iw_in, iw, 0)
                in_bounds = mask & ih_in & iw_in
                val = tl.load(x_ptr + row_base + safe_iw,
                              mask=in_bounds,
                              other=-float("inf"))
                max_val = tl.maximum(max_val, val)
        tl.store(y_ptr + offs, max_val, mask=mask)


class ModelNew(nn.Module):
    """Max Pooling 2D with Ascend-safe grid-capped Triton dispatch."""

    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 2,
                 padding: int = 1,
                 dilation: int = 3):
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)

    def _output_dim(self, L: int, k: int, s: int, p: int, d: int) -> int:
        eff_k = (k - 1) * d + 1
        return max((L + 2 * p - eff_k) // s + 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "Input must be 4D NCHW tensor"
        if x.device.type != "npu":
            return F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                                self.dilation)

        N, C, H, W = x.shape
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        DH = DW = self.dilation
        H_out = self._output_dim(H, KH, SH, PH, DH)
        W_out = self._output_dim(W, KW, SW, PW, DW)
        if H_out == 0 or W_out == 0:
            return x.new_empty((N, C, H_out, W_out))

        x_in = x.contiguous()
        y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
        total_out = N * C * H_out * W_out
        BLOCK = 256
        n_tiles = triton.cdiv(total_out, BLOCK)
        grid = (min(n_tiles, _MAX_GRID), )
        _maxpool2d_flat_kernel[grid](
            x_in,
            y,
            total_out,
            C,
            H,
            W,
            H_out,
            W_out,
            SH,
            SW,
            PH,
            PW,
            DH,
            DW,
            KH,
            KW,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=2,
        )
        return y


batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1


def max_pool2d_entry(x: torch.Tensor) -> torch.Tensor:
    return ModelNew(*get_init_inputs())(x)


def get_inputs():
    x = torch.rand(batch_size, channels, height, width, device="npu")
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
