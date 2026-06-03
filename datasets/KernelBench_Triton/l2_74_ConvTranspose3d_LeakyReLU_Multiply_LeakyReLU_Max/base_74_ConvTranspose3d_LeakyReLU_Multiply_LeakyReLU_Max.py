import torch
import torch.nn as nn
import triton
import triton.language as tl


DEFAULT_IN_CHANNELS = 16
DEFAULT_OUT_CHANNELS = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 1
DEFAULT_OUTPUT_PADDING = 1
DEFAULT_MULTIPLIER_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1, 1)
DEFAULT_BLOCK_H = 16
DEFAULT_BLOCK_W = 8
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 3


@triton.jit
def _fused_leaky_mul_maxpool3d_2x2x2(
    channel_idx_ptr,      # *i32 [C_WORK]
    x_ptr,                # *f32 [N, C, D, H, W]
    mult_ptr,             # *f32 [C, 1, 1, 1]
    y_ptr,                # *f32 [N, C, D//2, H//2, W//2]
    N, D, H, W,           # input sizes
    C_WORK,               # number of channels handled by this launch
    x_sN, x_sC, x_sD, x_sH, x_sW,  # x strides
    m_sC,                 # multiplier stride along C dim
    oD, oH, oW,           # output sizes
    y_sN, y_sC, y_sD, y_sH, y_sW,  # y strides
    h_tiles,              # number of H tiles
    w_tiles,              # number of tiles along W for grid axis-2 decomposition
    NEG_SLOPE: tl.constexpr,
    USE_MIN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program ids
    pid_nc = tl.program_id(0)      # ranges over N*C
    pid_d = tl.program_id(1)       # ranges over outD
    pid_hw = tl.program_id(2)      # ranges over h_tiles * w_tiles

    # Decode (n, c)
    n = pid_nc // C_WORK
    ci = pid_nc - n * C_WORK
    c = tl.load(channel_idx_ptr + ci)

    ht = pid_hw // w_tiles
    wt = pid_hw - ht * w_tiles

    ho = ht * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    wo = wt * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    mask_out = (ho < oH) & (wo < oW)

    # Corresponding input base indices (2x downsample window)
    d0 = 2 * pid_d
    h0 = 2 * ho
    w0_2 = 2 * wo

    # Base offsets (scalar base + vectorized W offsets)
    base_nc = n * x_sN + c * x_sC
    base_w = base_nc + d0 * x_sD + h0 * x_sH + w0_2 * x_sW

    # Per-channel multiplier (broadcast across spatial dims)
    m = tl.load(mult_ptr + c * m_sC)

    # Shorthand strides
    sD = x_sD
    sH = x_sH
    sW = x_sW

    # 8 neighbors of the 2x2x2 pooling window (vectorized along W)
    o000 = base_w
    o001 = base_w + sW
    o010 = base_w + sH
    o011 = o010 + sW
    o100 = base_w + sD
    o101 = o100 + sW
    o110 = o100 + sH
    o111 = o110 + sW

    # Loads: D/H/W are in-bounds when the output tile is valid.
    v000 = tl.load(x_ptr + o000, mask=mask_out, other=0.0)
    v001 = tl.load(x_ptr + o001, mask=mask_out, other=0.0)
    v010 = tl.load(x_ptr + o010, mask=mask_out, other=0.0)
    v011 = tl.load(x_ptr + o011, mask=mask_out, other=0.0)
    v100 = tl.load(x_ptr + o100, mask=mask_out, other=0.0)
    v101 = tl.load(x_ptr + o101, mask=mask_out, other=0.0)
    v110 = tl.load(x_ptr + o110, mask=mask_out, other=0.0)
    v111 = tl.load(x_ptr + o111, mask=mask_out, other=0.0)

    if USE_MIN:
        pooled0 = tl.minimum(v000, v001)
        pooled1 = tl.minimum(v010, v011)
        pooled2 = tl.minimum(v100, v101)
        pooled3 = tl.minimum(v110, v111)
        pooled4 = tl.minimum(pooled0, pooled1)
        pooled5 = tl.minimum(pooled2, pooled3)
        pooled = tl.minimum(pooled4, pooled5)
        vout = pooled * (m * NEG_SLOPE)
    else:
        pooled0 = tl.maximum(v000, v001)
        pooled1 = tl.maximum(v010, v011)
        pooled2 = tl.maximum(v100, v101)
        pooled3 = tl.maximum(v110, v111)
        pooled4 = tl.maximum(pooled0, pooled1)
        pooled5 = tl.maximum(pooled2, pooled3)
        pooled = tl.maximum(pooled4, pooled5)
        vout = tl.where(pooled >= 0, pooled * m, pooled * (m * NEG_SLOPE * NEG_SLOPE))

    # Store result
    out_base = n * y_sN + c * y_sC + pid_d * y_sD
    tl.store(y_ptr + out_base + ho * y_sH + wo * y_sW, vout, mask=mask_out)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter, 
    applies LeakyReLU again, and performs a max pooling operation.
    """
    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
        output_padding=DEFAULT_OUTPUT_PADDING,
        multiplier_shape=DEFAULT_MULTIPLIER_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        if x.device.type != "npu" or self.multiplier.device.type != "npu":
            raise RuntimeError("ModelNew expects NPU tensors so the Triton kernel path is exercised.")

        N, C, D, H, W = x.shape
        oD, oH, oW = D // 2, H // 2, W // 2
        y = torch.empty((N, C, oD, oH, oW), device=x.device, dtype=x.dtype)

        BLOCK_H = DEFAULT_BLOCK_H if oH >= DEFAULT_BLOCK_H else oH
        BLOCK_W = DEFAULT_BLOCK_W if oW >= DEFAULT_BLOCK_W else oW
        h_tiles = triton.cdiv(oH, BLOCK_H)
        w_tiles = triton.cdiv(oW, BLOCK_W)
        num_warps = DEFAULT_NUM_WARPS
        multiplier = self.multiplier.view(-1)
        pos_channels = torch.nonzero(multiplier >= 0, as_tuple=False).flatten().to(dtype=torch.int32)
        neg_channels = torch.nonzero(multiplier < 0, as_tuple=False).flatten().to(dtype=torch.int32)

        def launch(channel_idx, use_min):
            if channel_idx.numel() == 0:
                return
            grid = (N * channel_idx.numel(), oD, h_tiles * w_tiles)
            _fused_leaky_mul_maxpool3d_2x2x2[grid](
                channel_idx, x, self.multiplier, y,
                N, D, H, W,
                channel_idx.numel(),
                *x.stride(),
                self.multiplier.stride()[0],
                oD, oH, oW,
                *y.stride(),
                h_tiles=h_tiles,
                w_tiles=w_tiles,
                NEG_SLOPE=self.leaky_relu.negative_slope,
                USE_MIN=use_min,
                BLOCK_H=BLOCK_H,
                BLOCK_W=BLOCK_W,
                num_warps=num_warps,
                num_stages=DEFAULT_NUM_STAGES,
            )

        launch(pos_channels, False)
        launch(neg_channels, True)
        return y
batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]
