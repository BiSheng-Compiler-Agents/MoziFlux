import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

DEFAULT_IN_CHANNELS = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 0


@triton.jit
def _dwconv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    H_OUT,
    W_OUT,
    S: tl.constexpr,
    P: tl.constexpr,
    K: tl.constexpr,
    stride_xN,
    stride_xC,
    stride_xH,
    stride_xW,
    stride_wC,
    stride_wH,
    stride_wW,
    stride_yN,
    stride_yC,
    stride_yH,
    stride_yW,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    n_ch = pid_nc // C
    c = pid_nc % C

    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    ow = w_start + tl.arange(0, BLOCK_W)
    out_mask = ow < W_OUT

    y_base_nc = n_ch * stride_yN + c * stride_yC
    x_plane_base = n_ch * stride_xN + c * stride_xC
    w_base_c = c * stride_wC

    acc0 = tl.zeros([BLOCK_W], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_W], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_W], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_W], dtype=tl.float32)
    acc4 = tl.zeros([BLOCK_W], dtype=tl.float32)
    acc5 = tl.zeros([BLOCK_W], dtype=tl.float32)
    acc6 = tl.zeros([BLOCK_W], dtype=tl.float32)
    acc7 = tl.zeros([BLOCK_W], dtype=tl.float32)

    if (S == 1) & (P == 0):
        for kh in tl.static_range(0, K):
            w_row_base = w_base_c + kh * stride_wH
            for kw in tl.static_range(0, K):
                w_val = tl.load(w_ptr + w_row_base + kw * stride_wW)
                iw = ow + kw
                for oh_idx in tl.static_range(0, BLOCK_H):
                    oh = h_start + oh_idx
                    x_ptrs = x_ptr + x_plane_base + (oh + kh) * stride_xH + iw * stride_xW
                    x_val = tl.load(x_ptrs, mask=out_mask, other=0.0)
                    if oh_idx == 0: acc0 += x_val * w_val
                    elif oh_idx == 1: acc1 += x_val * w_val
                    elif oh_idx == 2: acc2 += x_val * w_val
                    elif oh_idx == 3: acc3 += x_val * w_val
                    elif oh_idx == 4: acc4 += x_val * w_val
                    elif oh_idx == 5: acc5 += x_val * w_val
                    elif oh_idx == 6: acc6 += x_val * w_val
                    elif oh_idx == 7: acc7 += x_val * w_val
    else:
        for kh in tl.static_range(0, K):
            w_row_base = w_base_c + kh * stride_wH
            iw0 = ow * S - P
            for kw in tl.static_range(0, K):
                w_val = tl.load(w_ptr + w_row_base + kw * stride_wW)
                valid_w = (iw0 + kw >= 0) & (iw0 + kw < W)
                for oh_idx in tl.static_range(0, BLOCK_H):
                    oh = h_start + oh_idx
                    ih0 = oh * S - P
                    ih = ih0 + kh
                    valid_h = (ih >= 0) & (ih < H)
                    mask = out_mask & valid_h & valid_w
                    x_ptrs = x_ptr + x_plane_base + ih * stride_xH + (iw0 + kw) * stride_xW
                    x_val = tl.load(x_ptrs, mask=mask, other=0.0)
                    if oh_idx == 0: acc0 += x_val * w_val
                    elif oh_idx == 1: acc1 += x_val * w_val
                    elif oh_idx == 2: acc2 += x_val * w_val
                    elif oh_idx == 3: acc3 += x_val * w_val
                    elif oh_idx == 4: acc4 += x_val * w_val
                    elif oh_idx == 5: acc5 += x_val * w_val
                    elif oh_idx == 6: acc6 += x_val * w_val
                    elif oh_idx == 7: acc7 += x_val * w_val

    if HAS_BIAS:
        b_val = tl.load(b_ptr + c)
        acc0 = acc0 + b_val
        acc1 = acc1 + b_val
        acc2 = acc2 + b_val
        acc3 = acc3 + b_val
        acc4 = acc4 + b_val
        acc5 = acc5 + b_val
        acc6 = acc6 + b_val
        acc7 = acc7 + b_val

    for oh_idx in tl.static_range(0, BLOCK_H):
        oh = h_start + oh_idx
        h_mask = oh < H_OUT
        y_ptrs = y_ptr + y_base_nc + oh * stride_yH + ow * stride_yW
        store_mask = out_mask & h_mask
        if oh_idx == 0: tl.store(y_ptrs, acc0, mask=store_mask)
        elif oh_idx == 1: tl.store(y_ptrs, acc1, mask=store_mask)
        elif oh_idx == 2: tl.store(y_ptrs, acc2, mask=store_mask)
        elif oh_idx == 3: tl.store(y_ptrs, acc3, mask=store_mask)
        elif oh_idx == 4: tl.store(y_ptrs, acc4, mask=store_mask)
        elif oh_idx == 5: tl.store(y_ptrs, acc5, mask=store_mask)
        elif oh_idx == 6: tl.store(y_ptrs, acc6, mask=store_mask)
        elif oh_idx == 7: tl.store(y_ptrs, acc7, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        bias: bool = False,
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise RuntimeError(
                "ModelNew expects Ascend NPU inputs; the Triton kernel path is the only supported runtime."
            )

        os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

        x = x.contiguous()
        w = self.conv2d.weight.contiguous()
        b = self.conv2d.bias
        if b is not None:
            b = b.contiguous()

        N_ch, C_ch, H_ch, W_ch = x.shape
        K_ch = w.shape[-1]
        S_ch = self.conv2d.stride[0]
        P_ch = self.conv2d.padding[0]

        H_OUT_ch = (H_ch + 2 * P_ch - K_ch) // S_ch + 1
        W_OUT_ch = (W_ch + 2 * P_ch - K_ch) // S_ch + 1

        y = torch.empty((N_ch, C_ch, H_OUT_ch, W_OUT_ch), device=x.device, dtype=x.dtype)

        stride_xN, stride_xC, stride_xH, stride_xW = x.stride()
        stride_wC, _, stride_wH, stride_wW = w.stride()
        stride_yN, stride_yC, stride_yH, stride_yW = y.stride()

        BLOCK_H = 8
        BLOCK_W = 320
        grid = (
            N_ch * C_ch,
            triton.cdiv(H_OUT_ch, BLOCK_H),
            triton.cdiv(W_OUT_ch, BLOCK_W),
        )

        num_warps = 4

        _dwconv2d_kernel[grid](
            x,
            w,
            (b if b is not None else y),
            y,
            N_ch,
            C_ch,
            H_ch,
            W_ch,
            H_OUT_ch,
            W_OUT_ch,
            S=S_ch,
            P=P_ch,
            K=K_ch,
            stride_xN=stride_xN,
            stride_xC=stride_xC,
            stride_xH=stride_xH,
            stride_xW=stride_xW,
            stride_wC=stride_wC,
            stride_wH=stride_wH,
            stride_wW=stride_wW,
            stride_yN=stride_yN,
            stride_yC=stride_yC,
            stride_yH=stride_yH,
            stride_yW=stride_yW,
            HAS_BIAS=(1 if b is not None else 0),
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
            num_stages=2,
        )
        return y


batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
