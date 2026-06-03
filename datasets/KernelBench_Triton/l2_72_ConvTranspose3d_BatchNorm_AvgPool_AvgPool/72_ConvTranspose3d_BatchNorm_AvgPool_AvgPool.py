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
DEFAULT_BIAS_SHAPE = (out_channels, 1, 1, 1)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_W": 32}, num_warps=2, num_stages=2),
    ],
    key=["OW"],
)
@triton.jit
def _avg_pool3d_k4s4_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_W: tl.constexpr,
):
    # Program ids:
    #  - axis 0 over (N * C * OD * OH)
    #  - axis 1 tiles over OW
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    # Decompose pid0 -> n, c, od, oh
    oh_idx = pid0 % OH
    tmp = pid0 // OH
    od_idx = tmp % OD
    tmp = tmp // OD
    nc_idx = tmp
    n_idx = nc_idx // C
    c_idx = nc_idx % C

    # Tile along W dimension
    w_out = pid1 * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_out < OW

    # Base pointers using strides
    x_base = (
        n_idx * stride_n
        + c_idx * stride_c
        + (od_idx * 4) * stride_d
        + (oh_idx * 4) * stride_h
    )
    y_base = (
        n_idx * out_stride_n
        + c_idx * out_stride_c
        + od_idx * out_stride_d
        + oh_idx * out_stride_h
    )

    # Each output corresponds to a 4x4x4 block in input with stride 4
    w_in_base = w_out * 4

    # Accumulate in FP32 for numerical stability
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    for kd in range(4):
        for kh in range(4):
            base_kdh = x_base + kd * stride_d + kh * stride_h
            for kw in range(4):
                ptrs = x_ptr + base_kdh + (w_in_base + kw) * stride_w
                vals = tl.load(ptrs, mask=w_mask, other=0.0).to(tl.float32)
                acc += vals

    # Average over 64 elements
    out_vals = acc * (1.0 / 64.0)

    # Store
    y_ptrs = y_ptr + y_base + w_out * out_stride_w
    tl.store(y_ptrs, out_vals, mask=w_mask)


def _avg_pool3d_k4s4_triton(x: torch.Tensor) -> torch.Tensor:
    # Fused two AvgPool3d(k=2, s=2) -> single AvgPool3d(k=4, s=4)
    if not _is_npu_tensor(x):
        raise RuntimeError("The fused AvgPool3d Triton wrapper expects an Ascend NPU tensor.")
    x = x.contiguous()
    N, C, D, H, W = x.shape
    # Two successive floor poolings with k=2, s=2 equal floor(D/4), etc.
    OD, OH, OW = D // 4, H // 4, W // 4
    y = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    if y.numel() == 0:
        return y

    sN, sC, sD, sH, sW = x.stride()
    osN, osC, osD, osH, osW = y.stride()

    # Launch kernel with autotuned BLOCK_W; grid depends on chosen meta
    grid = lambda META: (N * C * OD * OH, triton.cdiv(OW, META["BLOCK_W"]))
    _avg_pool3d_k4s4_kernel[grid](
        x, y,
        N, C, D, H, W,
        OD, OH, OW,
        sN, sC, sD, sH, sW,
        osN, osC, osD, osH, osW,
    )
    return y


class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization,
    then two average pooling layers. On CUDA, the two AvgPool3d(k=2, s=2) operations
    are fused into one Triton kernel implementing AvgPool3d with k=4, s=4 for speed.
    """
    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.bias_shape = bias_shape

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError(
                "ModelNew expects Ascend NPU inputs; the Triton kernel path is the only supported runtime."
            )
        if not _is_npu_tensor(self.conv_transpose.weight):
            raise RuntimeError("ModelNew weights must be moved to Ascend NPU before execution.")
        if self.conv_transpose.bias is not None and not _is_npu_tensor(self.conv_transpose.bias):
            raise RuntimeError("ModelNew bias must be moved to Ascend NPU before execution.")
        if not _is_npu_tensor(self.batch_norm.weight):
            raise RuntimeError("ModelNew batch-norm weights must be moved to Ascend NPU before execution.")
        if not _is_npu_tensor(self.batch_norm.bias):
            raise RuntimeError("ModelNew batch-norm bias must be moved to Ascend NPU before execution.")

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
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]