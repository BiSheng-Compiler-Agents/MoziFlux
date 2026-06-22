import torch
import torch.nn as nn
import triton
import triton.language as tl

DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 128


@triton.jit
def _pw_conv1x1_kernel(
    x_ptr,
    wt_ptr,
    bias_ptr,
    y_ptr,
    M,
    C_in,
    C_out,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C_out
    m_mask_i = m_mask.to(tl.int32)
    n_mask_i = n_mask.to(tl.int32)
    offs_m_safe = offs_m * m_mask_i
    offs_n_safe = offs_n * n_mask_i

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, C_in, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < C_in

        w_ptrs = wt_ptr + k_offs[:, None] * stride_wk + offs_n_safe[
            None, :] * stride_wn
        w_tile = tl.load(w_ptrs,
                         mask=(k_mask[:, None] & n_mask[None, :]),
                         other=0)

        x_ptrs = x_ptr + offs_m_safe[:, None] * stride_xm + k_offs[
            None, :] * stride_xk
        x_tile = tl.load(x_ptrs,
                         mask=(m_mask[:, None] & k_mask[None, :]),
                         other=0)

        acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_n_safe, mask=n_mask, other=0)
        acc += b[None, :].to(tl.float32)

    stride_ym = C_out
    stride_yn = 1
    y_ptrs = y_ptr + offs_m_safe[:, None] * stride_ym + offs_n_safe[
        None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=(m_mask[:, None] & n_mask[None, :]))


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation (1x1 Conv) using a Triton kernel on Ascend NPU.
    Semantics match nn.Conv2d with kernel_size=1.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=bias)
        self._cached_wt = None
        self._cached_bias = None
        self._cached_dtype = None
        self._cached_device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects Ascend NPU tensors.")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise RuntimeError(
                f"Unsupported dtype for Triton kernel: {x.dtype}")

        B, C_in, H, W = x.shape
        C_out = self.conv1d.out_channels
        M = B * H * W

        # Flat [M, C_in] view via as_strided: NCHW -> no-copy [B*H*W, C_in]
        x_flat = torch.as_strided(x, (M, C_in), (C_in, 1))

        # Output as flat [M, C_out]
        y_flat = torch.empty((M, C_out), device=x.device, dtype=x.dtype)

        # Cached weight/bias preparation
        if self._cached_wt is None or self._cached_dtype != x.dtype or self._cached_device != x.device:
            self._cached_wt = self.conv1d.weight.view(
                C_out, C_in).t().contiguous().to(device=x.device,
                                                 dtype=x.dtype)
            self._cached_bias = self.conv1d.bias.to(
                device=x.device,
                dtype=x.dtype) if self.conv1d.bias is not None else None
            self._cached_dtype = x.dtype
            self._cached_device = x.device
        wt = self._cached_wt
        bias = self._cached_bias
        has_bias = bias is not None

        stride_xm, stride_xk = x_flat.stride()
        stride_wk, stride_wn = wt.stride()

        BLOCK_M = 16
        BLOCK_N = 16
        BLOCK_K = 16

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C_out, BLOCK_N))

        _pw_conv1x1_kernel[grid](
            x_flat,
            wt,
            bias if has_bias else None,
            y_flat,
            M,
            C_in,
            C_out,
            stride_xm,
            stride_xk,
            stride_wk,
            stride_wn,
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        # Reshape output back to [B, C_out, H, W] (no contiguous copy)
        y = y_flat.view(B, H, W, C_out).permute(0, 3, 1, 2)
        return y


batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width, device='npu')
    return [x]


def get_init_inputs():
    return [in_channels, out_channels]
