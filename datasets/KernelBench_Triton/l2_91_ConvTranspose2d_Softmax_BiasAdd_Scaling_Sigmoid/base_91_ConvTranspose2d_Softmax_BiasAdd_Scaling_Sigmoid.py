import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softmax_bias_scale_sigmoid_1d(
    x_ptr,  # *f32, [N, C, H, W]
    bias_ptr,  # *f32, [C, 1, 1]
    y_ptr,  # *f32, [N, C, H, W]
    stride_n,
    stride_c,
    stride_h,
    stride_w,  # strides for NCHW
    b_stride_c,  # bias stride along C
    N,
    C,
    H,
    W,  # sizes
    scaling,  # float
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (n, h, w)
    pid = tl.program_id(axis=0)
    HW = H * W
    n = pid // HW
    hw = pid - n * HW
    h = hw // W
    w = hw - h * W

    base = n * stride_n + h * stride_h + w * stride_w

    c_idx = tl.arange(0, BLOCK_SIZE)
    mask = c_idx < C
    x_offsets = base + c_idx * stride_c

    # Load inputs; use L2-prefetch (cg) as this is streaming
    x = tl.load(x_ptr + x_offsets,
                mask=mask,
                other=-float("inf"),
                cache_modifier=".cg").to(tl.float32)

    # Stable softmax across channel dimension
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    denom = tl.sum(x_exp, axis=0)
    inv_denom = 1.0 / denom
    sm = x_exp * inv_denom

    # Bias add, scale, sigmoid (pre-scale bias and use FMA)
    s = tl.full((), scaling, tl.float32)
    b = tl.load(bias_ptr + c_idx * b_stride_c,
                mask=mask,
                other=0.0,
                cache_modifier=".ca").to(tl.float32)
    b_scaled = b * s
    z = tl.fma(sm, s, b_scaled)
    out = 1.0 / (1.0 + tl.exp(-z))
    tl.store(y_ptr + x_offsets, out, mask=mask)


@triton.jit
def _softmax_bias_scale_sigmoid_tiled_cw(
        x_ptr,  # *f32, [N, C, H, W]
        bias_ptr,  # *f32, [C, 1, 1]
        y_ptr,  # *f32, [N, C, H, W]
        stride_n,
        stride_c,
        stride_h,
        stride_w,  # strides for NCHW
        b_stride_c,  # bias stride along C
        N,
        C,
        H,
        W,  # sizes
        scaling,  # float
        BLOCK_C: tl.constexpr,  # tile in C (channels)
        BLOCK_W: tl.constexpr,  # tile in W (contiguous)
):
    # 2D grid:
    #  - axis 0: over (n, h) pairs
    #  - axis 1: over tiles of W
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    n = pid0 // H
    h = pid0 - n * H
    w_start = pid1 * BLOCK_W

    # Offsets in W (contiguous in memory)
    w_idx = w_start + tl.arange(0, BLOCK_W)
    w_mask = w_idx < W

    # Base (n, h) offset
    base = n * stride_n + h * stride_h

    # Channel tile indices
    c_idx = tl.arange(0, BLOCK_C)
    c_mask = c_idx < C

    # Pointers for a [BLOCK_C, BLOCK_W] tile
    ptrs = base + c_idx[:, None] * stride_c + w_idx[None, :] * stride_w
    # Load a full [C, Wtile] slab once into registers
    x_tile = tl.load(x_ptr + ptrs,
                     mask=c_mask[:, None] & w_mask[None, :],
                     other=-float("inf"),
                     cache_modifier=".cg").to(tl.float32)

    # Softmax along channel dimension for each column independently
    m = tl.max(x_tile, axis=0)  # [BLOCK_W]
    x_exp = tl.exp(x_tile - m[None, :])  # [BLOCK_C, BLOCK_W]
    denom = tl.sum(x_exp, axis=0)  # [BLOCK_W]
    inv_denom = 1.0 / denom
    sm = x_exp * inv_denom[None, :]  # [BLOCK_C, BLOCK_W]

    # Load bias once per channel and broadcast along W tile
    s = tl.full((), scaling, tl.float32)
    b = tl.load(bias_ptr + c_idx * b_stride_c,
                mask=c_mask,
                other=0.0,
                cache_modifier=".ca").to(tl.float32)  # [BLOCK_C]
    b_scaled = b * s
    z = tl.fma(sm, s, b_scaled[:, None])
    out = 1.0 / (1.0 + tl.exp(-z))

    # Store results using the same pointers
    tl.store(y_ptr + ptrs, out, mask=c_mask[:, None] & w_mask[None, :])


@triton.jit
def _softmax_bias_scale_sigmoid_tiled_chw(
    x_ptr,  # *f32, [N, C, H, W]
    bias_ptr,  # *f32, [C, 1, 1]
    y_ptr,  # *f32, [N, C, H, W]
    stride_n,
    stride_c,
    b_stride_c,
    N,
    C,
    HW,
    scaling,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    TILE_GROUP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    c_idx = tl.arange(0, BLOCK_C)
    c_mask = c_idx < C

    base = pid_n * stride_n
    s = tl.full((), scaling, tl.float32)
    b = tl.load(
        bias_ptr + c_idx * b_stride_c,
        mask=c_mask,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)
    b_scaled = (b * s)[:, None]

    for tile_idx in tl.static_range(0, TILE_GROUP):
        hw_start = (pid_hw * TILE_GROUP + tile_idx) * BLOCK_HW
        hw_idx = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_idx < HW

        ptrs = base + c_idx[:, None] * stride_c + hw_idx[None, :]
        x_tile = tl.load(
            x_ptr + ptrs,
            mask=c_mask[:, None] & hw_mask[None, :],
            other=-float("inf"),
            cache_modifier=".cg",
        ).to(tl.float32)

        m = tl.max(x_tile, axis=0)
        x_exp = tl.exp(x_tile - m[None, :])
        denom = tl.sum(x_exp, axis=0)
        z = tl.fma(x_exp, s / denom[None, :], b_scaled)
        out = tl.sigmoid(z)
        tl.store(y_ptr + ptrs, out, mask=c_mask[:, None] & hw_mask[None, :])


@triton.jit
def _softmax_bias_scale_sigmoid_tiled_chw_c128(
    x_ptr,  # *f32, [N, C, H, W]
    bias_ptr,  # *f32, [C, 1, 1]
    y_ptr,  # *f32, [N, C, H, W]
    stride_n,
    stride_c,
    b_stride_c,
    N,
    HW,
    scaling,
    BLOCK_HW: tl.constexpr,
    TILE_GROUP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    c_idx = tl.arange(0, 128)
    base = pid_n * stride_n
    s = tl.full((), scaling, tl.float32)
    b = tl.load(
        bias_ptr + c_idx * b_stride_c,
        cache_modifier=".ca",
    ).to(tl.float32)
    b_scaled = (b * s)[:, None]

    for tile_idx in tl.static_range(0, TILE_GROUP):
        hw_start = (pid_hw * TILE_GROUP + tile_idx) * BLOCK_HW
        hw_idx = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_idx < HW

        ptrs = base + c_idx[:, None] * stride_c + hw_idx[None, :]
        x_tile = tl.load(
            x_ptr + ptrs,
            mask=hw_mask[None, :],
            other=-float("inf"),
            cache_modifier=".cg",
        ).to(tl.float32)

        m = tl.max(x_tile, axis=0)
        x_exp = tl.exp(x_tile - m[None, :])
        denom = tl.sum(x_exp, axis=0)
        z = tl.fma(x_exp, s / denom[None, :], b_scaled)
        out = tl.sigmoid(z)
        tl.store(y_ptr + ptrs, out, mask=hw_mask[None, :])


# Keep the previous single-axis kernel for compatibility (not used in fast path)
@triton.jit
def _softmax_bias_scale_sigmoid_nchw(
    x_ptr,  # *f32, [N, C, H, W]
    bias_ptr,  # *f32, [C, 1, 1]
    y_ptr,  # *f32, [N, C, H, W]
    stride_n,
    stride_c,
    stride_h,
    stride_w,  # strides for NCHW
    b_stride_c,  # bias stride along C
    N,
    C: tl.constexpr,
    H,
    W,  # sizes (C constexpr to enable compile-time specialization)
    scaling,  # float
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (n, h, w)
    HW = H * W
    n = pid // HW
    hw = pid - n * HW
    h = hw // W
    w = hw - h * W

    # Base pointer offset for this (n, h, w) row across channels
    base = n * stride_n + h * stride_h + w * stride_w

    # Channel indices for the block
    c_idx = tl.arange(0, BLOCK_SIZE)
    mask = c_idx < C

    # Offsets for input/output along C
    x_offsets = base + c_idx * stride_c
    # Load input once; masked lanes get -inf so they don't affect reductions
    x = tl.load(x_ptr + x_offsets,
                mask=mask,
                other=-float("inf"),
                cache_modifier=".cg").to(tl.float32)

    # Stable softmax along C
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    denom = tl.sum(x_exp, axis=0)
    inv_denom = 1.0 / denom
    sm = x_exp * inv_denom

    # Load bias per channel
    b_offsets = c_idx * b_stride_c
    b = tl.load(bias_ptr + b_offsets,
                mask=mask,
                other=0.0,
                cache_modifier=".ca").to(tl.float32)

    # Scale and sigmoid (FMA)
    s = tl.full((), scaling, tl.float32)
    z = tl.fma(sm, s, b * s)
    out = 1.0 / (1.0 + tl.exp(-z))

    # Store result
    tl.store(y_ptr + x_offsets, out, mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << ((x - 1).bit_length())


def fused_softmax_bias_scale_sigmoid(x: torch.Tensor, bias: torch.Tensor,
                                     scaling_factor: float) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError(
            "fused_softmax_bias_scale_sigmoid requires an Ascend NPU tensor input"
        )
    if bias.device != x.device:
        raise RuntimeError(
            "bias must be placed on the same Ascend NPU device as x")

    x = x.contiguous()
    bias = bias.contiguous()

    N, C, H, W = x.shape
    HW = H * W
    y = torch.empty_like(x)

    stride_n, stride_c, stride_h, stride_w = x.stride()
    b_stride_c = bias.stride(0)

    # Prefer a flattened spatial tile so one program covers a contiguous HW slab.
    use_tiled = HW >= 8 and stride_h == W and stride_w == 1
    use_exact_c128 = use_tiled and C == 128

    if use_tiled:
        if HW >= 512:
            BLOCK_HW = 64
        elif HW >= 128:
            BLOCK_HW = 32
        elif HW >= 32:
            BLOCK_HW = 16
        else:
            BLOCK_HW = 8
        TILE_GROUP = 4 if HW >= 2048 else (4 if HW >= 256 else 2)
        grid = (N, triton.cdiv(HW, BLOCK_HW * TILE_GROUP))
        num_warps = 8 if BLOCK_HW >= 32 else 4
        if use_exact_c128:
            _softmax_bias_scale_sigmoid_tiled_chw_c128[grid](
                x,
                bias,
                y,
                stride_n,
                stride_c,
                b_stride_c,
                N,
                HW,
                float(scaling_factor),
                BLOCK_HW=BLOCK_HW,
                TILE_GROUP=TILE_GROUP,
                num_stages=4,
            )
        else:
            BLOCK_C = _next_power_of_2(C)
            _softmax_bias_scale_sigmoid_tiled_chw[grid](
                x,
                bias,
                y,
                stride_n,
                stride_c,
                b_stride_c,
                N,
                C,
                HW,
                float(scaling_factor),
                BLOCK_C=BLOCK_C,
                BLOCK_HW=BLOCK_HW,
                TILE_GROUP=TILE_GROUP,
                num_stages=4,
            )
    else:
        # 1D fallback across (n, h, w)
        BLOCK_SIZE = C  # exact size avoids masked lanes
        grid = (N * H * W, )
        if BLOCK_SIZE <= 64:
            num_warps = 2
        elif BLOCK_SIZE <= 256:
            num_warps = 4
        else:
            num_warps = 8
        _softmax_bias_scale_sigmoid_1d[grid](
            x,
            bias,
            y,
            stride_n,
            stride_c,
            stride_h,
            stride_w,
            b_stride_c,
            N,
            C,
            H,
            W,
            float(scaling_factor),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=3,
        )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies softmax, adds a bias term, scales the result, and applies sigmoid.
    The post-convolution ops are fused into a fast Triton kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_softmax_bias_scale_sigmoid(x, self.bias, self.scaling_factor)
        return x


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, stride, padding,
        output_padding, bias_shape, scaling_factor
    ]
