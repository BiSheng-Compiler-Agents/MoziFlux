import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softmax_sigmoid_fused_5d(
    x_ptr, y_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * D * H * W
    row_mask = pid < total_rows

    w_idx = pid % W
    tmp = pid // W
    h_idx = tmp % H
    tmp = tmp // H
    d_idx = tmp % D
    n_idx = tmp // D

    base = (n_idx * stride_n + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w).to(tl.int64)
    ch_offsets = tl.arange(0, BLOCK_C)

    m = -float("inf")
    c0 = 0
    while c0 < C:
        ch = c0 + ch_offsets
        ch_mask = ch < C
        ptrs = x_ptr + base + (ch * stride_c)
        x = tl.load(ptrs, mask=row_mask & ch_mask, other=-float("inf"))
        m = tl.maximum(m, tl.max(x.to(tl.float32), axis=0))
        c0 += BLOCK_C

    l = 0.0
    c0 = 0
    while c0 < C:
        ch = c0 + ch_offsets
        ch_mask = ch < C
        ptrs = x_ptr + base + (ch * stride_c)
        x = tl.load(ptrs, mask=row_mask & ch_mask, other=-float("inf")).to(tl.float32)
        l += tl.sum(tl.exp(x - m), axis=0)
        c0 += BLOCK_C

    inv_l = 1.0 / l

    c0 = 0
    while c0 < C:
        ch = c0 + ch_offsets
        ch_mask = ch < C
        ptrs = x_ptr + base + (ch * stride_c)
        x = tl.load(ptrs, mask=row_mask & ch_mask, other=-float("inf")).to(tl.float32)
        soft = tl.exp(x - m) * inv_l
        sig = 1.0 / (1.0 + tl.exp(-soft))
        out_ptrs = y_ptr + base + (ch * stride_c)
        tl.store(out_ptrs, sig, mask=row_mask & ch_mask)
        c0 += BLOCK_C


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """
    def __init__(
        self,
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        bias=True,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        if x.device.type != "npu":
            raise RuntimeError("ModelNew requires execution on Ascend NPU.")
        if x.dtype not in (torch.float16, torch.float32):
            raise RuntimeError(f"Unsupported dtype for fused Triton path: {x.dtype}")

        N, C, D, H, W = x.shape
        y = torch.empty_like(x)
        sN, sC, sD, sH, sW = x.stride()
        total_rows = N * D * H * W

        def grid(meta):
            return (total_rows,)

        _softmax_sigmoid_fused_5d[grid](
            x, y,
            N, C, D, H, W,
            sN, sC, sD, sH, sW,
            BLOCK_C=64,
        )
        return y
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]