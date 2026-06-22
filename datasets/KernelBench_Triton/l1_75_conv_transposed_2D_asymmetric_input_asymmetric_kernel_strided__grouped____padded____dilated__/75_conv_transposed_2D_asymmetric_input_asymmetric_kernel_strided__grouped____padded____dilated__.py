import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

DEFAULT_IN_CHANNELS = 32
DEFAULT_OUT_CHANNELS = 64
DEFAULT_KERNEL_SIZE = (3, 5)
DEFAULT_STRIDE = (2, 3)
DEFAULT_PADDING = (1, 2)
DEFAULT_DILATION = (2, 1)
DEFAULT_GROUPS = 4


@triton.jit
def _upsample_insert_zeros_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    H_UP,
    W_UP,
    STRIDE_H,
    STRIDE_W,
    in_strideN,
    in_strideC,
    in_strideH,
    in_strideW,
    out_strideN,
    out_strideC,
    out_strideH,
    out_strideW,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    hw_start = pid_hw * BLOCK_HW
    offs = hw_start + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)

    h_idx = offs // W
    w_idx = offs - h_idx * W

    x_base = x_ptr + n * in_strideN + c * in_strideC
    y_base = y_ptr + n * out_strideN + c * out_strideC

    vals = tl.load(x_base + h_idx * in_strideH + w_idx * in_strideW,
                   mask=mask,
                   other=0)

    ho = h_idx * STRIDE_H
    wo = w_idx * STRIDE_W

    tl.store(y_base + ho * out_strideH + wo * out_strideW, vals, mask=mask)


class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel,
    grouped, padded, and dilated.

    This implementation uses a Triton kernel to upsample (insert zeros) the input tensor and
    then computes the equivalent result via a standard conv2d with flipped weights:
      conv_transpose2d(x, w, stride, padding, dilation, groups)
    == conv2d(upsample(x, stride), flip(w).group_transpose(),
              stride=1, padding=dilation*(k-1)-padding, dilation=dilation, groups=groups)

    The delivered implementation is NPU-only and requires the transformed conv2d path
    to be valid for the configured parameters.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: tuple = DEFAULT_KERNEL_SIZE,
        stride: tuple = DEFAULT_STRIDE,
        padding: tuple = DEFAULT_PADDING,
        dilation: tuple = DEFAULT_DILATION,
        groups: int = DEFAULT_GROUPS,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _upsample_insert_zeros(self, x: torch.Tensor,
                               stride_hw: tuple[int, int]) -> torch.Tensor:
        # x: [N, C, H, W], upsample by stride_hw inserting zeros between elements.
        N, C, H, W = x.shape
        sH, sW = stride_hw
        H_up = (H - 1) * sH + 1
        W_up = (W - 1) * sW + 1

        # Zero-initialized output ensures sparsity structure is correct
        y = torch.zeros((N, C, H_up, W_up), device=x.device, dtype=x.dtype)

        # Ensure contiguous tensors to simplify stride math (units are elements, not bytes).
        x_contig = x.contiguous()
        y_contig = y  # already contiguous

        in_strides = x_contig.stride()
        out_strides = y_contig.stride()

        # Use a slightly larger tile and more warps to improve memory throughput
        BLOCK = 512
        grid = (N * C, triton.cdiv(H * W, BLOCK))
        _upsample_insert_zeros_kernel[grid](
            x_contig,
            y_contig,
            N,
            C,
            H,
            W,
            H_up,
            W_up,
            sH,
            sW,
            in_strides[0],
            in_strides[1],
            in_strides[2],
            in_strides[3],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            out_strides[3],
            BLOCK_HW=BLOCK,
            num_warps=8,
            num_stages=2,
        )
        return y_contig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        if x.device.type != "npu":
            raise RuntimeError("ModelNew requires Ascend NPU tensors.")

        # Extract parameters
        sH, sW = self.conv_transpose2d.stride
        pH, pW = self.conv_transpose2d.padding
        dH, dW = self.conv_transpose2d.dilation
        groups = self.conv_transpose2d.groups
        w = self.conv_transpose2d.weight.to(
            dtype=x.dtype)  # [Cin, Cout/groups, kH, kW]
        bias = self.conv_transpose2d.bias
        if bias is not None:
            bias = bias.to(dtype=x.dtype)

        kH, kW = w.shape[2], w.shape[3]

        # Compute equivalent conv2d padding
        pad_h2 = dH * (kH - 1) - pH
        pad_w2 = dW * (kW - 1) - pW

        if pad_h2 < 0 or pad_w2 < 0:
            raise RuntimeError(
                "Equivalent conv2d padding must be non-negative for this operator."
            )

        # Triton upsample (insert zeros)
        x_up = self._upsample_insert_zeros(x, (sH, sW))

        # Correct group-aware weight transform:
        # w: [Cin, Cout/G, kH, kW] -> flip spatial -> [G, Cin/G, Cout/G, kH, kW]
        # -> permute to [G, Cout/G, Cin/G, kH, kW] -> reshape to [Cout, Cin/G, kH, kW]
        G = groups
        Cin = w.shape[0]
        Cout_per_g = w.shape[1]
        Cin_per_g = Cin // G
        Cout = Cout_per_g * G

        w_flip = w.flip(dims=(2, 3)).contiguous()
        w_conv = (w_flip.view(G, Cin_per_g, Cout_per_g, kH,
                              kW).permute(0, 2, 1, 3,
                                          4).reshape(Cout, Cin_per_g, kH,
                                                     kW).contiguous())

        return F.conv2d(
            x_up,
            w_conv,
            bias=bias,
            stride=1,
            padding=(pad_h2, pad_w2),
            dilation=(dH, dW),
            groups=groups,
        )


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4


def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, stride, padding, dilation,
        groups
    ]
