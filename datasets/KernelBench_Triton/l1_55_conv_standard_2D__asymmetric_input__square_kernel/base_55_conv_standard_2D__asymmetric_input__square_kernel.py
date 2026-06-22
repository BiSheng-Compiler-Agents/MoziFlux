import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_row_major_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    tl.multiple_of(offs_m, 8)
    tl.multiple_of(offs_n, 8)

    k_start = 0
    while k_start < K:
        k_idx = k_start + offs_k
        a_ptrs = a_ptr + offs_m[:,
                                None] * stride_am + k_idx[None, :] * stride_ak
        b_ptrs = b_ptr + k_idx[:,
                               None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        tl.compile_hint(a, "dot_pad_only_k")
        tl.compile_hint(b, "dot_pad_only_k")
        acc += tl.dot(a, b)
        k_start += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _matmul_triton(a_2d: torch.Tensor, b_2d: torch.Tensor) -> torch.Tensor:
    if a_2d.dtype != torch.float32 or b_2d.dtype != torch.float32:
        raise RuntimeError(
            "matmul Triton path currently supports float32 only")

    a_c = a_2d.contiguous()
    b_c = b_2d.contiguous()
    m, k = a_c.shape
    k_b, n = b_c.shape
    if k_b != k:
        raise RuntimeError("matmul input shapes are incompatible")

    c = torch.empty((m, n), device=a_c.device, dtype=torch.float32)
    grid = (triton.cdiv(m, 128), triton.cdiv(n, 64))
    gemm_row_major_kernel[grid](
        a_c,
        b_c,
        c,
        m,
        n,
        k,
        a_c.stride(0),
        a_c.stride(1),
        b_c.stride(0),
        b_c.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return c


def _conv2d_triton_forward(x,
                           weight,
                           bias,
                           stride=1,
                           padding=0,
                           dilation=1,
                           groups=1):
    if x.device.type != "npu":
        raise RuntimeError("conv2d Triton kernel requires NPU inputs")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise RuntimeError(
            "conv2d Triton kernel currently supports float32 inputs and weights only"
        )
    if bias is not None and bias.dtype != torch.float32:
        raise RuntimeError(
            "conv2d Triton kernel currently supports float32 bias only")
    if groups != 1:
        raise RuntimeError(
            "conv2d Triton kernel currently supports groups=1 only")

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    n, c, h, w = x.shape
    oc, ci, kh, kw = weight.shape
    if ci != c:
        raise RuntimeError(
            "in_channels must match weight in_channels for groups=1")

    cols = F.unfold(
        x,
        kernel_size=(kh, kw),
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    # [N, D, L] -> [N * L, D]
    cols_2d = cols.transpose(1, 2).reshape(-1, cols.shape[1]).contiguous()
    weight_2d = weight.reshape(oc, -1).transpose(0, 1).contiguous()

    out_2d = _matmul_triton(cols_2d, weight_2d)
    if bias is not None:
        out_2d = out_2d + bias.view(1, oc)

    h_out = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    w_out = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    return out_2d.view(n, h_out * w_out,
                       oc).transpose(1, 2).reshape(n, oc, h_out, w_out)


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _conv2d_triton_forward(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups,
        )


batch_size = 8
height = 512
width = 1024
in_channels = 64
out_channels = 128
kernel_size = 3


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
