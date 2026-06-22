import torch
import torch.nn as nn

import triton
import triton.language as tl

DEFAULT_IN_CHANNELS = 128
DEFAULT_OUT_CHANNELS = 128
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 0


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_H': 16,
            'BLOCK_W': 256
        },
                      num_warps=8,
                      num_stages=2),
    ],
    key=['H_OUT', 'W_OUT'],
)
@triton.jit
def _dwconv2d_kernel(
        x_ptr,  # *const T, [N, C_in, H_in, W_in]
        w_ptr,  # *const T, [C_out, 1, K, K]
        b_ptr,  # *const T or nullptr if no bias, [C_out]
        y_ptr,  # *mut T,   [N, C_out, H_out, W_out]
        N: tl.constexpr,
        C_IN,
        C_OUT,
        H_IN,
        W_IN,
        H_OUT,
        W_OUT,
        STRIDE,
        PADDING,
        OCPG,  # out_channels per group (= out_channels // in_channels)
        K: tl.constexpr,  # kernel size (square)
        HAS_BIAS: tl.constexpr,  # compile-time flag
        BLOCK_H: tl.constexpr,  # tile size along H_out
        BLOCK_W: tl.constexpr,  # tile size along W_out
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    # Decompose pid across (N, C_OUT)
    n = pid_nc // C_OUT
    oc = pid_nc % C_OUT
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H_OUT

    # Tile of output width this program computes
    w_start = pid_w * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    w_mask = w_offsets < W_OUT

    # Map output channel to its input channel (depthwise groups = in_channels)
    ic = oc // OCPG

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    # Precompute base scales for pointer arithmetic (int64 to avoid overflow)
    n = tl.full((), n, tl.int64)
    oc_i64 = tl.full((), oc, tl.int64)
    ic_i64 = tl.full((), ic, tl.int64)
    C_IN = tl.full((), C_IN, tl.int64)
    C_OUT = tl.full((), C_OUT, tl.int64)
    H_IN = tl.full((), H_IN, tl.int64)
    W_IN = tl.full((), W_IN, tl.int64)
    H_OUT = tl.full((), H_OUT, tl.int64)
    W_OUT_i64 = tl.full((), W_OUT, tl.int64)
    STRIDE = tl.full((), STRIDE, tl.int32)
    PADDING = tl.full((), PADDING, tl.int32)

    w_offsets_i64 = w_offsets.to(tl.int64)
    h_offsets_i64 = h_offsets.to(tl.int64)
    wi_base = w_offsets * STRIDE - PADDING
    hi_base = h_offsets * STRIDE - PADDING
    x_nc_base = (n * C_IN + ic_i64) * H_IN * W_IN
    w_oc_base = oc_i64 * (K * K)

    for r in range(K):
        hi = hi_base + r
        hi_mask = h_mask & (hi >= 0) & (hi < H_IN.to(tl.int32))
        hi_offsets = hi.to(tl.int64)[:, None] * W_IN

        for s in range(K):
            wi = wi_base + s
            in_bounds_w = (wi >= 0) & (wi < W_IN.to(tl.int32))
            mask = hi_mask[:, None] & w_mask[None, :] & in_bounds_w[None, :]
            x_offsets = x_nc_base + hi_offsets + wi.to(tl.int64)[None, :]
            x_vals = tl.load(x_ptr + x_offsets, mask=mask,
                             other=0).to(tl.float32)

            w_offset = w_oc_base + (r * K + s)
            w_val = tl.load(w_ptr + w_offset).to(tl.float32)
            acc += x_vals * w_val

    if HAS_BIAS:
        b_val = tl.load(b_ptr + oc_i64).to(tl.float32)
        acc += b_val

    y_base = (n * C_OUT + oc_i64) * H_OUT * W_OUT_i64
    y_offsets = y_base + h_offsets_i64[:, None] * W_OUT_i64 + w_offsets_i64[
        None, :]
    tl.store(y_ptr + y_offsets, acc, mask=h_mask[:, None] & w_mask[None, :])


def _depthwise_conv2d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "The Triton depthwise conv wrapper expects an Ascend NPU input tensor."
        )
    if not _is_npu_tensor(weight):
        raise RuntimeError(
            "The Triton depthwise conv wrapper expects NPU weights.")
    if bias is not None and not _is_npu_tensor(bias):
        raise RuntimeError(
            "The Triton depthwise conv wrapper expects NPU bias tensors.")

    if isinstance(stride, tuple):
        if len(stride) != 2 or stride[0] != stride[1]:
            raise ValueError("Only symmetric stride values are supported.")
        stride = stride[0]
    if isinstance(padding, tuple):
        if len(padding) != 2 or padding[0] != padding[1]:
            raise ValueError("Only symmetric padding values are supported.")
        padding = padding[0]

    # Expect weight of shape [C_out, 1, K, K] for groups=in_channels
    C_out, C_per_group, K, K2 = weight.shape
    if K != K2:
        raise ValueError("Only square kernels are supported.")
    C_in = x.shape[1]
    if C_per_group != 1 or (C_out % C_in) != 0:
        raise ValueError(
            "Weight must have shape [C_out, 1, K, K] with C_out divisible by C_in."
        )

    N, _, H_in, W_in = x.shape
    ocpg = C_out // C_in

    # Output dims (no dilation)
    H_out = (H_in + 2 * padding - K) // stride + 1
    W_out = (W_in + 2 * padding - K) // stride + 1

    # Ensure contiguous
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous() if bias is not None else None

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Launch kernel
    block_h = 16
    grid = (triton.cdiv(W_out, 256), triton.cdiv(H_out, block_h), N * C_out)
    has_bias = bias is not None

    _dwconv2d_kernel[grid](
        x_c,
        w_c,
        (b_c if has_bias else x_c),  # dummy ptr if no bias, not used
        y,
        N,
        C_in,
        C_out,
        H_in,
        W_in,
        H_out,
        W_out,
        stride,
        padding,
        ocpg,
        K=K,
        HAS_BIAS=has_bias,
    )
    return y


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        # Keep PyTorch Conv2d to hold parameters with identical initialization
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        if not _is_npu_tensor(x):
            raise RuntimeError(
                "ModelNew expects Ascend NPU inputs; the Triton kernel path is the only supported runtime."
            )
        if not _is_npu_tensor(self.conv2d.weight):
            raise RuntimeError(
                "ModelNew weights must be moved to Ascend NPU before execution."
            )
        if self.conv2d.bias is not None and not _is_npu_tensor(
                self.conv2d.bias):
            raise RuntimeError(
                "ModelNew bias must be moved to Ascend NPU before execution.")

        return _depthwise_conv2d_triton(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
        )


batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
width_in = 512
height_in = 256
stride = 1
padding = 0


def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
