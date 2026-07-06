import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _add_bias_expand_kernel(
    tmp_ptr,        # *fp32, contiguous [N, W]
    bias_ptr,       # *fp32, contiguous [B, 1, 1]
    y_ptr,          # *fp32, contiguous [N, B, 1, W]
    N: tl.constexpr,
    BIAS_C: tl.constexpr,
    W: tl.constexpr,
    sbc,
    syn,
    syc,
    syh,
    syw,
    BLOCK_W: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w_blk = tl.program_id(1)
    pid_b_blk = tl.program_id(2)

    w_offsets = pid_w_blk * BLOCK_W + tl.arange(0, BLOCK_W)
    b_offsets = pid_b_blk * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_w = w_offsets < W
    mask_b = b_offsets < BIAS_C

    vals = tl.load(tmp_ptr + pid_n * W + w_offsets, mask=mask_w, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + b_offsets * sbc, mask=mask_b, other=0.0).to(tl.float32)
    out = vals[None, :] + bias[:, None]
    out_ptrs = y_ptr + pid_n * syn + b_offsets[:, None] * syc + w_offsets[None, :] * syw
    tl.store(out_ptrs, out, mask=mask_b[:, None] & mask_w[None, :])


class ModelNew(nn.Module):
    """ConvTranspose2d -> min over channel -> sum over height -> GELU -> add bias."""

    def __init__(
        self,
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        bias_shape=(16, 1, 1),
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x).contiguous()
        reduced = x.min(dim=1, keepdim=True).values.sum(dim=2, keepdim=True)
        gelu_vals = F.gelu(reduced)

        # CPU/non-fp32 path uses PyTorch broadcasting. The NPU fp32 path keeps the
        # expensive ConvTranspose/min/sum/GELU on ACL and uses a tiny Triton epilogue
        # only for the final broadcast add.
        if gelu_vals.dtype != torch.float32 or gelu_vals.device.type != "npu":
            return gelu_vals + self.bias

        N, _, _, W = gelu_vals.shape
        bias_c = self.bias.shape[0]
        y = torch.empty((N, bias_c, 1, W), device=gelu_vals.device, dtype=gelu_vals.dtype)
        tmp = gelu_vals.reshape(N, W).contiguous()
        b_c = self.bias.contiguous()

        BLOCK_W = 64
        BLOCK_B = 32
        grid = (N, triton.cdiv(W, BLOCK_W), triton.cdiv(bias_c, BLOCK_B))
        _add_bias_expand_kernel[grid](
            tmp,
            b_c,
            y,
            N,
            bias_c,
            W,
            b_c.stride(0),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            y.stride(3),
            BLOCK_W=BLOCK_W,
            BLOCK_B=BLOCK_B,
            num_warps=4,
            num_stages=2,
        )
        return y


batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ]
