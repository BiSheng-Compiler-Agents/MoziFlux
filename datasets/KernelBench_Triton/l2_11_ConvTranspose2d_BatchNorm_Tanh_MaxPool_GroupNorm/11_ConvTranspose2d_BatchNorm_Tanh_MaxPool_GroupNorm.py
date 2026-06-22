import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 512
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 128
DEFAULT_KERNEL_SIZE = 5
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 1
DEFAULT_GROUPS = 8
DEFAULT_NUM_GROUPS = 8
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


@triton.jit
def _tanh_maxpool2x2_nchw_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    BLOCK_WO: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_ho = tl.program_id(1)
    pid_wo = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C

    ho_offsets = pid_ho * BLOCK_HO + tl.arange(0, BLOCK_HO)
    wo_offsets = pid_wo * BLOCK_WO + tl.arange(0, BLOCK_WO)

    ho_mask = ho_offsets < H_OUT
    wo_mask = wo_offsets < W_OUT
    HO = ho_offsets[:, None]
    WO = wo_offsets[None, :]
    out_mask = ho_mask[:, None] & wo_mask[None, :]

    base_x = (n * C + c) * (H * W)
    base_y = (n * C + c) * (H_OUT * W_OUT)
    y_offs = base_y + HO * W_OUT + WO

    ih0 = HO * 2
    iw0 = WO * 2
    row0 = base_x + ih0 * W
    row1 = row0 + W

    offs00 = row0 + iw0
    offs01 = offs00 + 1
    offs10 = row1 + iw0
    offs11 = offs10 + 1

    v00 = tl.load(x_ptr + offs00, mask=out_mask, other=-float("inf"))
    v01 = tl.load(x_ptr + offs01, mask=out_mask, other=-float("inf"))
    v10 = tl.load(x_ptr + offs10, mask=out_mask, other=-float("inf"))
    v11 = tl.load(x_ptr + offs11, mask=out_mask, other=-float("inf"))

    # Pool over the already-activated tensor.
    m0 = tl.maximum(v00, v01)
    m1 = tl.maximum(v10, v11)
    mp = tl.maximum(m0, m1)
    tl.store(y_ptr + y_offs, mp, mask=out_mask)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    Tanh + MaxPool2d are fused into a single Triton kernel for improved performance on GPU.
    """

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
        groups=DEFAULT_GROUPS,
        num_groups=DEFAULT_NUM_GROUPS,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.group_norm = nn.GroupNorm(num_groups=num_groups,
                                       num_channels=out_channels)

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in {torch.float32, torch.bfloat16}:
            raise RuntimeError(
                f"ModelNew supports only float32 and bfloat16 inputs, got {x.dtype}"
            )

        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = torch.tanh(x)
        x = x.contiguous()
        N, C, H, W = x.shape
        H_OUT = H // 2
        W_OUT = W // 2
        y = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)
        block_ho = 8
        block_wo = 32
        grid = (N * C, triton.cdiv(H_OUT,
                                   block_ho), triton.cdiv(W_OUT, block_wo))
        _tanh_maxpool2x2_nchw_kernel[grid](x,
                                           y,
                                           N=N,
                                           C=C,
                                           H=H,
                                           W=W,
                                           H_OUT=H_OUT,
                                           W_OUT=W_OUT,
                                           BLOCK_HO=block_ho,
                                           BLOCK_WO=block_wo,
                                           num_warps=4,
                                           num_stages=1)
        x = y

        x = self.group_norm(x)
        return x


batch_size = 512
in_channels = 64
out_channels = 128
height = width = 2048
kernel_size = 5
stride = 1
padding = 1
groups = 8
num_groups = 8
height, width = 32, 32


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, stride, padding, groups,
        num_groups
    ]
