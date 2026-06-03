import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        if A_ROW_MAJOR:
            a_offsets = m_offsets[:, None] * K + k_offsets[None, :]
        else:
            a_offsets = m_offsets[:, None] + k_offsets[None, :] * M
        a_mask = m_mask[:, None] & k_mask[None, :]
        a_tile = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        b_offsets = k_offsets[:, None] * N + n_offsets[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]
        b_tile = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)
    c_offsets = m_offsets[:, None] * N + n_offsets[None, :]
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def conv2d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    if x.device.type != "npu":
        raise ValueError(f"conv2d_triton requires an Ascend NPU tensor, got {x.device.type!r}")
    if groups != 1:
        raise NotImplementedError("conv2d_triton only supports groups=1")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported input dtype: {x.dtype}")

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    B, CI, H, W = x.shape
    CO, CI_w, KH, KW = weight.shape
    if CI != CI_w:
        raise ValueError("Input channels mismatch")

    STRH, STRW = stride
    PADH, PADW = padding
    DILH, DILW = dilation
    OH = (H + 2 * PADH - DILH * (KH - 1) - 1) // STRH + 1
    OW = (W + 2 * PADW - DILW * (KW - 1) - 1) // STRW + 1
    if OH <= 0 or OW <= 0:
        raise ValueError("Invalid output shape computed for conv2d_triton")

    Kdim = CI * KH * KW
    Mdim = B * OH * OW
    Ndim = CO

    out_dtype = x.dtype

    # im2col on NPU via unfold
    x_c = x.contiguous()
    x_unfold = F.unfold(x_c, kernel_size=(KH, KW), dilation=(DILH, DILW),
                         padding=(PADH, PADW), stride=(STRH, STRW))
    # x_unfold: (B, Kdim, OH*OW) -> (Mdim, Kdim) row-major
    x_mat = x_unfold.permute(0, 2, 1).reshape(Mdim, Kdim).contiguous()

    # Weight: (CO, Kdim) -> (Kdim, CO) for left-matmul B
    w_mat = weight.reshape(CO, Kdim).t().contiguous()

    y_flat = torch.empty((Mdim, Ndim), device=x.device, dtype=out_dtype)

    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_K = 256
    NUM_STAGES = 2

    def grid(meta):
        return (triton.cdiv(Mdim, meta["BLOCK_M"]), triton.cdiv(Ndim, meta["BLOCK_N"]))

    _gemm_kernel[grid](
        x_mat, w_mat, y_flat,
        M=Mdim, N=Ndim, K=Kdim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        A_ROW_MAJOR=True,
        num_warps=8, num_stages=NUM_STAGES,
    )

    # Reshape back to NCHW (already in out_dtype, no final cast needed)
    y = y_flat.reshape(B, OH, OW, CO).permute(0, 3, 1, 2).contiguous()

    if bias is not None:
        y += bias.view(1, CO, 1, 1).to(y.dtype)
    return y


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return conv2d_triton(
            x=x,
            weight=self.conv2d.weight,
            bias=self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups,
        )
batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
