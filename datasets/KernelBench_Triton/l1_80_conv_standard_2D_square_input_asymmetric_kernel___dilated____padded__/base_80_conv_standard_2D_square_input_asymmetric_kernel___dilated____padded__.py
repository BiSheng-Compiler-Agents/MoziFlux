import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_nn_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:,
                                None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:,
                                None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        acc = tl.dot(a, b, acc, input_precision='hf32')
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _conv2d_triton_nchw(x, weight, bias, stride, padding, dilation):
    if x.device.type != 'npu' or weight.device.type != 'npu':
        raise AssertionError('Triton kernel requires NPU tensors')
    N, C, H, W = x.shape
    OC, Cw, KH, KW = weight.shape
    assert C == Cw, 'Input channels mismatch'
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    H_OUT = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    W_OUT = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    K = C * KH * KW
    w_packed = weight.permute(1, 2, 3,
                              0).reshape(K, OC).contiguous().to(torch.float32)
    b_ = (bias.contiguous().to(torch.float32) if bias is not None else
          torch.zeros(OC, device=x.device, dtype=torch.float32))
    y = torch.empty((N, OC, H_OUT, W_OUT),
                    device=x.device,
                    dtype=torch.float32)
    for n in range(N):
        xn = x[n:n + 1].contiguous().to(torch.float32)
        patches = torch.nn.functional.unfold(xn,
                                             kernel_size=(KH, KW),
                                             dilation=(dh, dw),
                                             padding=(ph, pw),
                                             stride=(sh, sw))
        patches = patches.squeeze(0).T.contiguous()
        M_val = H_OUT * W_OUT
        out_flat = torch.empty((M_val, OC),
                               device=x.device,
                               dtype=torch.float32)
        grid = (triton.cdiv(M_val, 256), triton.cdiv(OC, 64))
        gemm_nn_kernel[grid](patches, w_packed, out_flat, M_val, OC, K,
                             patches.stride(0), patches.stride(1),
                             w_packed.stride(0), w_packed.stride(1),
                             out_flat.stride(0), out_flat.stride(1), 256, 64,
                             192)
        y[n:n + 1] = out_flat.T.reshape(1, OC, H_OUT, W_OUT) + b_.reshape(
            1, OC, 1, 1)
    return y.to(dtype=x.dtype) if x.dtype != torch.float32 else y


def conv2d_triton_nchw(x, weight, bias=None, stride=1, padding=0, dilation=1):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return _conv2d_triton_nchw(x, weight, bias, stride, padding, dilation)


class ModelNew(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=(0, 0),
                 dilation=(1, 1),
                 bias=False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)

    def forward(self, x):
        weight = self.conv2d.weight
        bias = self.conv2d.bias
        stride = self.conv2d.stride if isinstance(
            self.conv2d.stride, tuple) else (self.conv2d.stride,
                                             self.conv2d.stride)
        padding = self.conv2d.padding if isinstance(
            self.conv2d.padding, tuple) else (self.conv2d.padding,
                                              self.conv2d.padding)
        dilation = self.conv2d.dilation if isinstance(
            self.conv2d.dilation, tuple) else (self.conv2d.dilation,
                                               self.conv2d.dilation)
        return conv2d_triton_nchw(x, weight, bias, stride, padding, dilation)


batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512
stride = 1
padding = (2, 4)
dilation = (2, 3)


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
