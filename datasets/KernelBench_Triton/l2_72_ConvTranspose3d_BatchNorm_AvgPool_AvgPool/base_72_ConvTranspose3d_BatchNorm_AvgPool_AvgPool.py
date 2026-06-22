import torch
import torch.nn as nn
import triton
import triton.language as tl

DEFAULT_IN_CHANNELS = 3
DEFAULT_OUT_CHANNELS = 16
DEFAULT_DEPTH = 32
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 1
DEFAULT_BIAS_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1, 1)

POOL_BLOCK_D = 5
POOL_BLOCK_H = 15
POOL_BLOCK_W = 16
POOL_NUM_WARPS = 4
POOL_NUM_STAGES = 2
POOL_VECTORIZE_KW = True
POOL_COMPILE_HINTS = False


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _avg_pool3d_k4s4_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    stride_n,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    VECTORIZE_KW: tl.constexpr,
    COMPILE_HINTS: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    oh_tiles = tl.cdiv(OH, BLOCK_H)
    od_tiles = tl.cdiv(OD, BLOCK_D)
    oh_tile = pid0 % oh_tiles
    tmp = pid0 // oh_tiles
    od_tile = tmp % od_tiles
    nc_idx = tmp // od_tiles

    n_idx = nc_idx // C
    c_idx = nc_idx % C

    od_idx = od_tile * BLOCK_D + tl.arange(0, BLOCK_D)[:, None, None]
    oh_idx = oh_tile * BLOCK_H + tl.arange(0, BLOCK_H)[None, :, None]
    w_out = pid1 * BLOCK_W + tl.arange(0, BLOCK_W)[None, None, :]

    od_mask = od_idx < OD
    oh_mask = oh_idx < OH
    w_mask = w_out < OW
    mask = od_mask & oh_mask & w_mask

    x_base = (n_idx * stride_n + c_idx * stride_c + (od_idx * 4) * stride_d +
              (oh_idx * 4) * stride_h)
    y_base = (n_idx * out_stride_n + c_idx * out_stride_c +
              od_idx * out_stride_d + oh_idx * out_stride_h)
    w_in_base = w_out * 4
    if COMPILE_HINTS:
        w_in_base = tl.max_contiguous(tl.multiple_of(w_in_base, (1, 1, 4)),
                                      (1, 1, BLOCK_W))

    acc = tl.zeros([BLOCK_D, BLOCK_H, BLOCK_W], dtype=tl.float32)
    if VECTORIZE_KW:
        kw = tl.arange(0, 4)[None, None, None, :]
        for kd in range(4):
            for kh in range(4):
                ptrs = x_ptr + x_base[:, :, :,
                                      None] + kd * stride_d + kh * stride_h + (
                                          w_in_base[:, :, :, None] +
                                          kw) * stride_w
                vals = tl.load(ptrs, mask=mask[:, :, :, None],
                               other=0.0).to(tl.float32)
                acc += tl.sum(vals, axis=3)
    else:
        for kd in range(4):
            for kh in range(4):
                base = x_base + kd * stride_d + kh * stride_h
                for kw in range(4):
                    ptrs = x_ptr + base + (w_in_base + kw) * stride_w
                    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
                    acc += vals

    out_vals = acc * (1.0 / 64.0)
    y_ptrs = y_ptr + y_base + w_out * out_stride_w
    tl.store(y_ptrs, out_vals, mask=mask)


def _avg_pool3d_k4s4_triton(x: torch.Tensor) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "The fused AvgPool3d Triton wrapper expects an Ascend NPU tensor.")
    x = x.contiguous()
    N, C, D, H, W = x.shape
    OD, OH, OW = D // 4, H // 4, W // 4
    y = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    if y.numel() == 0:
        return y

    sN, sC, sD, sH, sW = x.stride()
    osN, osC, osD, osH, osW = y.stride()

    def grid(META):
        return (
            N * C * triton.cdiv(OD, META["BLOCK_D"]) *
            triton.cdiv(OH, META["BLOCK_H"]),
            triton.cdiv(OW, META["BLOCK_W"]),
        )

    _avg_pool3d_k4s4_kernel[grid](
        x,
        y,
        N,
        C,
        D,
        H,
        W,
        OD,
        OH,
        OW,
        sN,
        sC,
        sD,
        sH,
        sW,
        osN,
        osC,
        osD,
        osH,
        osW,
        BLOCK_D=POOL_BLOCK_D,
        BLOCK_H=POOL_BLOCK_H,
        BLOCK_W=POOL_BLOCK_W,
        VECTORIZE_KW=POOL_VECTORIZE_KW,
        COMPILE_HINTS=POOL_COMPILE_HINTS,
        num_warps=POOL_NUM_WARPS,
        num_stages=POOL_NUM_STAGES,
    )
    return y


class ModelNew(nn.Module):

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.bias_shape = bias_shape

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError(
                "ModelNew expects Ascend NPU inputs; the Triton kernel path is the only supported runtime."
            )
        if not _is_npu_tensor(self.conv_transpose.weight):
            raise RuntimeError(
                "ModelNew weights must be moved to Ascend NPU before execution."
            )
        if self.conv_transpose.bias is not None and not _is_npu_tensor(
                self.conv_transpose.bias):
            raise RuntimeError(
                "ModelNew bias must be moved to Ascend NPU before execution.")
        if not _is_npu_tensor(self.batch_norm.weight):
            raise RuntimeError(
                "ModelNew batch-norm weights must be moved to Ascend NPU before execution."
            )
        if not _is_npu_tensor(self.batch_norm.bias):
            raise RuntimeError(
                "ModelNew batch-norm bias must be moved to Ascend NPU before execution."
            )

        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        return _avg_pool3d_k4s4_triton(x)


batch_size = 64
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, stride, padding, bias_shape
    ]
