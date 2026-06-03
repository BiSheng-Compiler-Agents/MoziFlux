import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_s1p0_vecoc_kernel(
        x_ptr,  # *f32
        w_ptr,  # *f32
        b_ptr,  # *f32 or dummy
        y_ptr,  # *f32
        N,  # int32 runtime
        H,  # int32 runtime
        W,  # int32 runtime
        OC,  # int32 runtime
        H_out,  # int32 runtime
        W_out,  # int32 runtime
        TILES_WO,  # int32 runtime: number of tiles along width
        C: tl.constexpr,  # compile-time in_channels
        K: tl.constexpr,  # compile-time kernel_size (square)
        BIAS: tl.constexpr,  # 0/1 compile-time whether to add bias
        BLOCK_HO: tl.constexpr,  # tile size along output height
        BLOCK_WO: tl.constexpr,  # tile size along output width
        BLOCK_OC: tl.
    constexpr,  # number of output channels computed per program
):
    pid_n = tl.program_id(axis=0)
    pid_ob = tl.program_id(axis=1)  # output-channel block id
    pid_tile = tl.program_id(axis=2)

    # Decode spatial tile index
    tile_ho = pid_tile // TILES_WO
    tile_wo = pid_tile % TILES_WO

    ho_offsets = tile_ho * BLOCK_HO + tl.arange(0, BLOCK_HO)
    wo_offsets = tile_wo * BLOCK_WO + tl.arange(0, BLOCK_WO)
    OH = ho_offsets[:, None]  # [BH, 1]
    OW = wo_offsets[None, :]  # [1, BW]

    mask_spatial = (OH < H_out) & (OW < W_out)

    # Strides for NCHW contiguous layout
    x_n_stride = C * H * W
    x_c_stride = H * W
    x_h_stride = W
    x_w_stride = 1

    y_n_stride = OC * H_out * W_out
    y_oc_stride = H_out * W_out
    y_h_stride = W_out
    y_w_stride = 1

    # Output channels this program computes
    oc_offsets = pid_ob * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = oc_offsets < OC

    # Base pointers for this batch
    x_base = pid_n * x_n_stride
    y_base = pid_n * y_n_stride

    # Accumulator: [BLOCK_OC, BLOCK_HO, BLOCK_WO]
    acc = tl.zeros((BLOCK_OC, BLOCK_HO, BLOCK_WO), dtype=tl.float32)

    # Direct convolution: y[n, oc, oh, ow] = sum_{c, kh, kw} x[n, c, oh+kh, ow+kw] * w[oc, c, kh, kw]
    for kh in range(0, K):
        for kw in range(0, K):
            ih = OH + kh
            iw = OW + kw
            x_hw_offsets = x_base + ih * x_h_stride + iw * x_w_stride  # [BH, BW]
            # For stride=1, padding=0, dilation=1, these are always in-bounds, but keep mask for safety
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            for c in range(0, C):
                # Load weights for a vector of OC
                w_off = ((oc_offsets * C + c) * K + kh) * K + kw  # [BLOCK_OC]
                w_vec = tl.load(w_ptr + w_off, mask=mask_oc,
                                other=0.0).to(tl.float32)  # [BLOCK_OC]
                # Load input tile for this (c, kh, kw)
                x_offs_c = x_hw_offsets + c * x_c_stride  # [BH, BW]
                x_val = tl.load(x_ptr + x_offs_c,
                                mask=mask_spatial & in_bounds,
                                other=0.0).to(tl.float32)  # [BH, BW]
                # FMA with broadcasting over OC
                acc += w_vec[:, None, None] * x_val[None, :, :]

    if BIAS:
        b_vec = tl.load(b_ptr + oc_offsets, mask=mask_oc,
                        other=0.0).to(tl.float32)  # [BLOCK_OC]
        acc += b_vec[:, None, None]

    # Store results
    y_offsets = (y_base + oc_offsets[:, None, None] * y_oc_stride +
                 OH[None, :, :] * y_h_stride + OW[None, :, :] * y_w_stride)
    store_mask = mask_oc[:, None, None] & mask_spatial[None, :, :]
    tl.store(y_ptr + y_offsets, acc, mask=store_mask)


def _triton_conv2d_s1p0(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    # Preconditions: stride=1, padding=0, dilation=1, groups=1, NCHW contiguous
    if x.device.type != "npu" or weight.device.type != "npu":
        raise ValueError("conv2d Triton path requires NPU tensors")
    if x.dtype != weight.dtype:
        raise ValueError("input and weight dtypes must match")
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("expected x and weight to be 4D tensors")
    N, C, H, W = x.shape
    OC, Cw, K, Kw = weight.shape
    if C != Cw or K != Kw:
        raise ValueError("only square kernels with matching input channels are supported")
    H_out = H - K + 1
    W_out = W - K + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError("invalid output spatial size")
    if bias is not None:
        if bias.device.type != "npu":
            raise ValueError("bias must be placed on NPU when provided")
        if bias.dtype != x.dtype:
            raise ValueError("bias dtype must match input dtype")
        if bias.ndim != 1 or bias.shape[0] != OC:
            raise ValueError("bias must be a 1D tensor with shape [out_channels]")

    # Allocate output (compute in fp32 for stability, cast back later)
    y = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=torch.float32)

    # Tiling configuration: fuse OC in block to reuse input tile
    BLOCK_HO = 4
    BLOCK_WO = 128
    BLOCK_OC = 16

    tiles_ho = triton.cdiv(H_out, BLOCK_HO)
    tiles_wo = triton.cdiv(W_out, BLOCK_WO)
    grid = (N, triton.cdiv(OC, BLOCK_OC), tiles_ho * tiles_wo)

    # Ensure contiguous memory
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous() if (bias is not None) else torch.empty(1, device=x.device, dtype=torch.float32)

    conv2d_nchw_s1p0_vecoc_kernel[grid](
        x_c, w_c, b_c, y,
        N, H, W, OC, H_out, W_out, tiles_wo,
        C=C, K=K, BIAS=1 if bias is not None else 0,
        BLOCK_HO=BLOCK_HO, BLOCK_WO=BLOCK_WO, BLOCK_OC=BLOCK_OC,
        num_warps=8, num_stages=2,
    )
    # Match input dtype
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    return y


def conv2d_standard_2d_square_input_square_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _triton_conv2d_s1p0(x, weight, bias)


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep a reference PyTorch module for parameter management and fallback
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        if x.device.type != "npu":
            raise ValueError("ModelNew.forward requires an NPU input tensor")
        if self.conv2d.stride != (1, 1):
            raise NotImplementedError("only stride=1 is supported")
        if self.conv2d.padding != (0, 0):
            raise NotImplementedError("only padding=0 is supported")
        if self.conv2d.dilation != (1, 1):
            raise NotImplementedError("only dilation=1 is supported")
        if self.conv2d.groups != 1:
            raise NotImplementedError("only groups=1 is supported")
        if x.requires_grad:
            raise NotImplementedError("autograd is not supported for this Triton path")

        w = self.conv2d.weight
        b = self.conv2d.bias
        if w.device.type != "npu":
            raise ValueError("model weights must be placed on NPU before calling forward")
        if b is not None and b.device.type != "npu":
            raise ValueError("model bias must be placed on NPU before calling forward")
        if w.dtype != x.dtype:
            w = w.to(dtype=x.dtype)
        if b is not None and b.dtype != x.dtype:
            b = b.to(dtype=x.dtype)
        return conv2d_standard_2d_square_input_square_kernel(x, w, b)
batch_size = 16
in_channels = 16
out_channels = 128
kernel_size = 3
width = 1024
height = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization