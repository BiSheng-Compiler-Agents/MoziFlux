import math
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _conv1d_fwd_kernel(
    x_ptr,           # float*        [N, IC, L_IN]
    w_ptr,           # float*        [OC, IC, K]
    b_ptr,           # float* or dummy (unused when HAS_BIAS=False) [OC]
    y_ptr,           # float*        [N, OC, L_OUT]
    N,               # int
    L_IN,            # int
    OC,              # int
    L_OUT,           # int
    x_stride_n,      # int
    x_stride_c,      # int
    x_stride_l,      # int
    w_stride_o,      # int
    w_stride_c,      # int
    w_stride_k,      # int
    y_stride_n,      # int
    y_stride_o,      # int
    y_stride_l,      # int
    STRIDE: tl.constexpr,     # int (constexpr)
    DILATION: tl.constexpr,   # int (constexpr)
    IC: tl.constexpr,         # int (constexpr)
    K: tl.constexpr,          # int (constexpr)
    HAS_BIAS: tl.constexpr,   # bool (constexpr)
    BLOCK_OC: tl.constexpr,   # tile size for OC
):
    # Program IDs over (N * L_OUT) and OC tiles (unchanged PID logic)
    pid_nl = tl.program_id(0)
    pid_ob = tl.program_id(1)

    n = pid_nl // L_OUT
    p = pid_nl % L_OUT

    # Tile of output channels handled by this program
    oc_offsets = pid_ob * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < OC

    # Accumulator for [BLOCK_OC] results in FP32
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # Base positions/pointers
    pos0 = p * STRIDE
    x_n_base = n * x_stride_n

    # Precompute base pointer per output-channel in the weights
    w_co_base = w_ptr + oc_offsets * w_stride_o

    # Reduction over input channels and kernel taps (no masked RBLOCK to reduce overhead)
    for ic in tl.static_range(0, IC):
        x_nc_base = x_ptr + x_n_base + ic * x_stride_c
        w_c_base = w_co_base + ic * w_stride_c
        for k in tl.static_range(0, K):
            t = pos0 + k * DILATION
            in_bounds = t < L_IN  # keep boundary check as required
            # Load single input value
            x_val = tl.load(x_nc_base + t * x_stride_l, mask=in_bounds, other=0.0)
            # Load vector of weights for this (ic, k) across BLOCK_OC output channels
            w_vec = tl.load(w_c_base + k * w_stride_k, mask=oc_mask, other=0.0)
            # FMA accumulate
            acc += w_vec * x_val

    # Add bias if present
    if HAS_BIAS:
        b_vec = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0)
        acc += b_vec

    # Store results
    y_idx = n * y_stride_n + oc_offsets * y_stride_o + p * y_stride_l
    tl.store(y_ptr + y_idx, acc, mask=oc_mask)


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use PyTorch Conv1d parameters for initialization parity
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        conv1d = self.conv1d
        if conv1d.groups != 1:
            raise RuntimeError("ModelNew only supports groups=1.")
        if conv1d.padding[0] != 0:
            raise RuntimeError("ModelNew only supports padding=0.")
        return conv_standard_1d_dilated_strided(
            x,
            conv1d.weight,
            conv1d.bias,
            stride=conv1d.stride[0],
            dilation=conv1d.dilation[0],
        )


def conv_standard_1d_dilated_strided(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    dilation: int = 1,
) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("conv_standard_1d_dilated_strided expects x on NPU.")
    if weight.device.type != "npu":
        raise RuntimeError("conv_standard_1d_dilated_strided expects weight on NPU.")
    if bias is not None and bias.device.type != "npu":
        raise RuntimeError("conv_standard_1d_dilated_strided expects bias on NPU.")
    if x.ndim != 3:
        raise RuntimeError(f"Expected x to have shape [N, IC, L_IN], got {tuple(x.shape)}.")
    if weight.ndim != 3:
        raise RuntimeError(
            f"Expected weight to have shape [OC, IC, K], got {tuple(weight.shape)}."
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise RuntimeError(
            f"Expected bias to have shape [{weight.shape[0]}], got {tuple(bias.shape)}."
        )
    if x.shape[1] != weight.shape[1]:
        raise RuntimeError(
            f"Input channels ({x.shape[1]}) must match weight.shape[1] ({weight.shape[1]})."
        )
    if stride < 1 or dilation < 1:
        raise RuntimeError(f"Invalid convolution parameters: stride={stride}, dilation={dilation}.")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if x.dtype not in supported_dtypes:
        raise RuntimeError(f"Unsupported input dtype: {x.dtype}.")
    if weight.dtype != x.dtype:
        raise RuntimeError(
            f"Expected weight dtype {x.dtype} to match input dtype, got {weight.dtype}."
        )
    if bias is not None and bias.dtype != x.dtype:
        raise RuntimeError(
            f"Expected bias dtype {x.dtype} to match input dtype, got {bias.dtype}."
        )

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None

    N, IC, L_IN = x.shape
    OC, IC_w, K = weight.shape
    if IC != IC_w:
        raise RuntimeError(f"Input channels mismatch: x has {IC}, weight expects {IC_w}.")
    receptive_field = dilation * (K - 1) + 1
    L_OUT = (L_IN - receptive_field) // stride + 1
    if L_OUT <= 0:
        raise RuntimeError(
            "Invalid output length; check input length, kernel_size, stride, and dilation."
        )

    y = torch.empty((N, OC, L_OUT), device=x.device, dtype=x.dtype)
    x_stride_n, x_stride_c, x_stride_l = x.stride()
    w_stride_o, w_stride_c, w_stride_k = weight.stride()
    y_stride_n, y_stride_o, y_stride_l = y.stride()

    block_oc = 64
    grid = (N * L_OUT, triton.cdiv(OC, block_oc))
    bias_ptr = bias if bias is not None else y.new_empty(1)

    _conv1d_fwd_kernel[grid](
        x,
        weight,
        bias_ptr,
        y,
        N,
        L_IN,
        OC,
        L_OUT,
        x_stride_n,
        x_stride_c,
        x_stride_l,
        w_stride_o,
        w_stride_c,
        w_stride_k,
        y_stride_n,
        y_stride_o,
        y_stride_l,
        STRIDE=stride,
        DILATION=dilation,
        IC=IC,
        K=K,
        HAS_BIAS=(bias is not None),
        BLOCK_OC=block_oc,
        num_warps=4,
        num_stages=2,
    )
    return y
batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = 3
length = 524280
stride = 3
dilation = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]