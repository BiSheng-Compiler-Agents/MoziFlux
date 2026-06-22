import torch
import torch.nn as nn
import triton
import triton.language as tl

DEFAULT_IN_CHANNELS = 8
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 0
DEFAULT_DILATION = 1


@triton.jit
def _dw_conv_kh1_kernel(
    x_ptr,  # *const T, [B, C, H_in, W_in]
    w_ptr,  # *const T, [C, 1, K, 1]
    b_ptr,  # *const T or nullptr, [C]
    y_ptr,  # *T, [B, C, H_out, W_out]
    B,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    K,
    S_h,
    S_w,
    P_h,
    P_w,
    D_h,
    D_w,
    x_bs,
    x_cs,
    x_hs,
    x_ws,
    w_cs,
    w_khs,
    y_bs,
    y_cs,
    y_hs,
    y_ws,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_bc // C
    c = pid_bc % C

    oh_ids = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow_ids = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_mask = (oh_ids[:, None] < H_out) & (ow_ids[None, :] < W_out)

    # Compute corresponding input column indices for kW=1 and validity
    iw0 = ow_ids * S_w - P_w
    valid_w = (iw0[None, :] >= 0) & (iw0[None, :] < W_in)

    # Base pointers for (b, c)
    x_bc_ptr = x_ptr + b * x_bs + c * x_cs
    y_bc_ptr = y_ptr + b * y_bs + c * y_cs

    # Precompute base offsets along width for the tile to reduce address arithmetic
    x_col_base = x_bc_ptr + iw0[None, :] * x_ws

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Double-buffered software pipelining across K dimension
    # Precompute base ih for kh=0
    oh_base = oh_ids * S_h - P_h

    # Handle K >= 1
    if K > 0:
        ih0 = oh_base + 0 * D_h
        valid_h0 = (ih0[:, None] >= 0) & (ih0[:, None] < H_in)
        mask0 = out_mask & valid_h0 & valid_w
        x_ptrs0 = x_col_base + ih0[:, None] * x_hs
        x_vals0 = tl.load(x_ptrs0, mask=mask0, other=0.0).to(tl.float32)
        w_base_ptr = w_ptr + c * w_cs
        w0 = tl.load(w_base_ptr + 0 * w_khs).to(tl.float32)

        # Main pipelined loop
        for kh in range(0, K - 1):
            khn = kh + 1
            ih1 = oh_base + khn * D_h
            valid_h1 = (ih1[:, None] >= 0) & (ih1[:, None] < H_in)
            mask1 = out_mask & valid_h1 & valid_w
            x_ptrs1 = x_col_base + ih1[:, None] * x_hs
            x_vals1 = tl.load(x_ptrs1, mask=mask1, other=0.0).to(tl.float32)
            w1 = tl.load(w_base_ptr + khn * w_khs).to(tl.float32)

            # FMA accumulate current buffer
            acc += x_vals0 * w0

            # Rotate buffers
            x_vals0 = x_vals1
            w0 = w1

        # Final buffered step
        acc += x_vals0 * w0

    if HAS_BIAS:
        b_val = tl.load(b_ptr + c).to(tl.float32)
        acc += b_val

    # Store results (cast handled by tl.store to y_ptr dtype)
    y_ptrs = y_bc_ptr + oh_ids[:, None] * y_hs + ow_ids[None, :] * y_ws
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def _cdiv(a, b):
    return (a + b - 1) // b


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


def _depthwise_conv2d_kh1_triton(x: torch.Tensor, weight: torch.Tensor,
                                 bias: torch.Tensor | None,
                                 stride: int | tuple, padding: int | tuple,
                                 dilation: int | tuple):
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

    # Normalize params to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    S_h, S_w = stride
    P_h, P_w = padding
    D_h, D_w = dilation

    B, C, H_in, W_in = x.shape
    # weight: [C, 1, K, 1]
    K = weight.shape[2]

    # Output sizes (PyTorch Conv2d formula)
    H_out = (H_in + 2 * P_h - D_h * (K - 1) - 1) // S_h + 1
    # For kW=1
    W_out = (W_in + 2 * P_w - D_w * (1 - 1) - 1) // S_w + 1

    y = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Ensure contiguous memory
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous() if bias is not None else None
    y_c = y  # output will be written directly

    x_bs, x_cs, x_hs, x_ws = x_c.stride()
    y_bs, y_cs, y_hs, y_ws = y_c.stride()
    # Weight strides (C, 1, K, 1)
    w_strides = w_c.stride()
    w_cs = w_strides[0]
    w_khs = w_strides[2]

    # Tuned tile sizes for better width-coalesced access on Hopper
    BLOCK_H = 32
    BLOCK_W = 128

    grid = (B * C, _cdiv(W_out, BLOCK_W), _cdiv(H_out, BLOCK_H))
    has_bias = bias is not None

    _dw_conv_kh1_kernel[grid](
        x_c,
        w_c,
        (b_c
         if has_bias else x_c),  # pass dummy ptr if no bias (won't be used)
        y_c,
        B,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        K,
        S_h,
        S_w,
        P_h,
        P_w,
        D_h,
        D_w,
        x_bs,
        x_cs,
        x_hs,
        x_ws,
        w_cs,
        w_khs,
        y_bs,
        y_cs,
        y_hs,
        y_ws,
        HAS_BIAS=has_bias,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=8,
        num_stages=3,
    )
    return y


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        dilation: int = DEFAULT_DILATION,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        # Keep a PyTorch Conv2d parameter container to match initialization/semantics exactly.
        self.conv2d = nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=(kernel_size, 1),
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=in_channels,
                                bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return _depthwise_conv2d_kh1_triton(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.conv2d.stride,
            self.conv2d.padding,
            self.conv2d.dilation,
        )


batch_size = 64
in_channels = 8
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0
dilation = 1


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]
