import torch
import torch.nn as nn

import triton
import triton.language as tl


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


def _can_use_default_coblock_path(
    *,
    bias: torch.Tensor | None,
    cin: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> bool:
    return (bias is None and cin == 32 and kernel_size == 3 and stride == 2
            and padding == 1 and dilation == 2)


@triton.jit
def conv_transpose1d_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    CIN,
    COUT,
    LIN,
    LOUT,
    K,
    STRIDE,
    PADDING,
    DILATION,
    stride_xn,
    stride_xc,
    stride_xl,
    stride_wci,
    stride_wco,
    stride_wk,
    stride_yn,
    stride_yc,
    stride_yl,
    HAS_BIAS: tl.constexpr,
    CIN_C: tl.constexpr,
    K_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    # Map program id to (n, co)
    co = pid0 % COUT
    n = pid0 // COUT

    # Offsets in output length dimension
    offs_t = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < LOUT

    # Base pointers
    y_base = y_ptr + n * stride_yn + co * stride_yc
    base_xn = x_ptr + n * stride_xn
    base_wco = w_ptr + co * stride_wco

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_T, ), dtype=tl.float32)

    # Optional bias
    if HAS_BIAS:
        b_val = tl.load(b_ptr + co)
        acc += b_val.to(tl.float32)

    # Precompute output->input mapping terms once
    i_base = offs_t + PADDING
    q = i_base // STRIDE
    r = i_base - q * STRIDE  # r = i_base % STRIDE

    # Loop over kernel taps and input channels (fully unrolled)
    for k in tl.static_range(0, K_C):
        kd = k * DILATION
        a = kd % STRIDE
        b = kd // STRIDE

        # Valid alignment when r == a
        mask_align = (r == a) & mask_t
        # Corresponding input index
        i_vec = q - b
        mask_i = mask_align & (i_vec >= 0) & (i_vec < LIN)
        # Safe offsets for loads
        i_safe = tl.where(mask_i, i_vec, 0)
        x_offs = i_safe * stride_xl

        wk_base = base_wco + k * stride_wk

        for ci in tl.static_range(0, CIN_C):
            # Load weight scalar w[ci, co, k]
            w_val = tl.load(wk_base + ci * stride_wci).to(tl.float32)

            # Load x[n, ci, i] for vector i (gather)
            x_base_ci = base_xn + ci * stride_xc
            x_vals = tl.load(x_base_ci + x_offs, mask=mask_i,
                             other=0.0).to(tl.float32)

            acc += x_vals * w_val

    # Store results
    tl.store(y_base + offs_t * stride_yl, acc, mask=mask_t)


@triton.jit
def conv_transpose1d_fwd_kernel_default_coblock(
    x_ptr,
    w_ptr,
    y_ptr,
    LIN,
    COUT,
    ACTIVE_LEN,
    stride_xn,
    stride_xc,
    stride_xl,
    stride_wci,
    stride_wco,
    stride_wk,
    stride_yn,
    stride_yc,
    stride_yl,
    BLOCK_T: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    CIN_C: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_co = tl.program_id(axis=1)
    pid_t = tl.program_id(axis=2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_t = offs_t < ACTIVE_LEN
    mask_co = offs_co < COUT
    offs_out = offs_t * 2 + 1

    x_base = x_ptr + pid_n * stride_xn
    y_base = y_ptr + pid_n * stride_yn
    acc = tl.zeros((BLOCK_T, BLOCK_CO), dtype=tl.float32)

    for k in tl.static_range(0, 3):
        i_idx = offs_t + 1 - k
        mask_i = mask_t & (i_idx >= 0) & (i_idx < LIN)
        i_safe = tl.where(mask_i, i_idx, 0)
        x_offs = i_safe * stride_xl
        wk_base = w_ptr + offs_co * stride_wco + k * stride_wk

        for ci in tl.static_range(0, CIN_C):
            x_vals = tl.load(x_base + ci * stride_xc + x_offs,
                             mask=mask_i,
                             other=0.0).to(tl.float32)
            w_vals = tl.load(wk_base + ci * stride_wci,
                             mask=mask_co,
                             other=0.0).to(tl.float32)
            acc += x_vals[:, None] * w_vals[None, :]

    y_ptrs = y_base + offs_co[None, :] * stride_yc + offs_out[:,
                                                              None] * stride_yl
    tl.store(y_ptrs, acc, mask=mask_t[:, None] & mask_co[None, :])


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        dilation: int = 2,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        # Keep a reference nn.ConvTranspose1d module for exact parameter initialization semantics
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects an Ascend NPU input tensor")

        # Extract parameters and ensure contiguity
        weight = self.conv1d_transpose.weight
        bias = self.conv1d_transpose.bias
        stride = self.conv1d_transpose.stride[0] if isinstance(
            self.conv1d_transpose.stride, tuple) else int(
                self.conv1d_transpose.stride)
        padding = self.conv1d_transpose.padding[0] if isinstance(
            self.conv1d_transpose.padding, tuple) else int(
                self.conv1d_transpose.padding)
        dilation = self.conv1d_transpose.dilation[0] if isinstance(
            self.conv1d_transpose.dilation, tuple) else int(
                self.conv1d_transpose.dilation)
        output_padding = 0  # matches the original constructor behavior

        if not _is_npu_tensor(weight):
            raise RuntimeError("ModelNew expects Ascend NPU weights")
        if bias is not None and not _is_npu_tensor(bias):
            raise RuntimeError("ModelNew expects Ascend NPU bias parameters")
        if x.dim() != 3:
            raise ValueError(
                f"expected a 3D input tensor, got shape {tuple(x.shape)}")
        if x.dtype not in (torch.float16, torch.float32):
            raise TypeError(f"unsupported input dtype: {x.dtype}")
        if weight.dtype != x.dtype:
            raise TypeError(
                f"weight dtype {weight.dtype} must match input dtype {x.dtype}"
            )
        if bias is not None and bias.dtype != x.dtype:
            raise TypeError(
                f"bias dtype {bias.dtype} must match input dtype {x.dtype}")

        weight = weight.contiguous()
        x = x.contiguous()

        N, Cin, Lin = x.shape
        Cin_w, Cout, K = weight.shape
        assert Cin == Cin_w, "Input channels mismatch"
        # PyTorch ConvTranspose1d output length formula
        Lout = (Lin - 1) * stride - 2 * padding + dilation * (
            K - 1) + output_padding + 1

        # Strides (element-wise) for pointer arithmetic
        stride_xn, stride_xc, stride_xl = x.stride()
        stride_wci, stride_wco, stride_wk = weight.stride()
        if _can_use_default_coblock_path(
                bias=bias,
                cin=Cin,
                kernel_size=K,
                stride=stride,
                padding=padding,
                dilation=dilation,
        ):
            y = torch.zeros((N, Cout, Lout), device=x.device, dtype=x.dtype)
            stride_yn, stride_yc, stride_yl = y.stride()
            active_len = (Lout + 1) // 2
            block_t = 32
            block_co = 8
            grid = (N, triton.cdiv(Cout,
                                   block_co), triton.cdiv(active_len, block_t))

            conv_transpose1d_fwd_kernel_default_coblock[grid](
                x,
                weight,
                y,
                Lin,
                Cout,
                active_len,
                stride_xn,
                stride_xc,
                stride_xl,
                stride_wci,
                stride_wco,
                stride_wk,
                stride_yn,
                stride_yc,
                stride_yl,
                BLOCK_T=block_t,
                BLOCK_CO=block_co,
                CIN_C=Cin,
                num_warps=4,
                num_stages=2,
            )
        else:
            y = torch.empty((N, Cout, Lout), device=x.device, dtype=x.dtype)
            stride_yn, stride_yc, stride_yl = y.stride()

            BLOCK_T = 128
            grid = (N * Cout, triton.cdiv(Lout, BLOCK_T))

            conv_transpose1d_fwd_kernel[grid](
                x,
                weight,
                bias if bias is not None else y,
                y,
                N,
                Cin,
                Cout,
                Lin,
                Lout,
                K,
                stride,
                padding,
                dilation,
                stride_xn,
                stride_xc,
                stride_xl,
                stride_wci,
                stride_wco,
                stride_wk,
                stride_yn,
                stride_yc,
                stride_yl,
                HAS_BIAS=1 if bias is not None else 0,
                CIN_C=Cin,
                K_C=K,
                BLOCK_T=BLOCK_T,
                num_warps=4,
                num_stages=2,
            )

        return y


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
length = 131072
stride = 2
padding = 1
dilation = 2


def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
