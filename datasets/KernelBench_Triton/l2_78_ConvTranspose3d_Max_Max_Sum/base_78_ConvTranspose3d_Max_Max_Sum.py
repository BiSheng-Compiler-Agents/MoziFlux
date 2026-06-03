import torch
import torch.nn as nn
import triton
import triton.language as tl


DEFAULT_BATCH_SIZE = 16
DEFAULT_IN_CHANNELS = 32
DEFAULT_OUT_CHANNELS = 64
DEFAULT_DEPTH = 32
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_KERNEL_SIZE = 5
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 2


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 2, 'BLOCK_H': 10, 'BLOCK_W': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 2, 'BLOCK_H': 10, 'BLOCK_W': 10}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 4, 'BLOCK_H': 10, 'BLOCK_W': 5}, num_warps=4, num_stages=2),
    ],
    key=['H2', 'W2'],
)
@triton.jit
def _maxpool_6x_3d_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    D2, H2, W2,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_ndc = tl.program_id(2)

    d_block = pid_ndc % tl.cdiv(D2, BLOCK_D)
    nc = pid_ndc // tl.cdiv(D2, BLOCK_D)
    n = nc // C
    c = nc % C
    d_out = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    md = d_out < D2

    # Tile of output H/W
    w_out = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_out = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mw = w_out < W2
    mh = h_out < H2
    m_hw = mh[:, None] & mw[None, :]

    d_start = d_out * 6
    h_start = h_out * 6
    w_start = w_out * 6

    m = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), -float('inf'), dtype=tl.float32)
    base_nc = n * stride_n + c * stride_c
    kw = tl.arange(0, 6)
    w_offsets = (w_start[:, None] + kw[None, :]) * stride_w
    valid_dhw6 = md[:, None, None, None] & m_hw[None, :, :, None]

    for kd in range(6):
        d_idx_off = (d_start + kd)[:, None, None, None] * stride_d
        for kh in range(6):
            h_offsets = (h_start + kh) * stride_h
            ptr = (
                x_ptr
                + base_nc
                + d_idx_off
                + h_offsets[None, :, None, None]
                + w_offsets[None, None, :, :]
            )
            val = tl.load(ptr, mask=valid_dhw6, other=-float('inf'))
            m = tl.maximum(m, tl.max(val.to(tl.float32), axis=3))

    out_base = out_ptr + n * out_stride_n + c * out_stride_c + d_out[:, None, None] * out_stride_d
    out_ptrs = out_base + h_out[None, :, None] * out_stride_h + w_out[None, None, :] * out_stride_w
    tl.store(out_ptrs, m, mask=md[:, None, None] & m_hw[None, :, :])


def _fused_two_pools_into_one(x: torch.Tensor) -> torch.Tensor:
    """
    Replace consecutive MaxPool3d(kernel=2, stride=2) and MaxPool3d(kernel=3, stride=3)
    with a single 3D max-pool using kernel=6, stride=6 (equivalent composition).
    Implemented as a Triton kernel that keeps per-channel outputs.
    """
    if not _is_npu_tensor(x):
        raise RuntimeError("The fused MaxPool3d Triton wrapper expects an Ascend NPU tensor.")
    if x.ndim != 5:
        raise ValueError(f"expected a 5D tensor, got shape {tuple(x.shape)}")

    x = x.contiguous()
    N, C, D, H, W = x.shape
    if D < 6 or H < 6 or W < 6:
        raise ValueError(
            "input spatial dimensions must all be at least 6 to compose MaxPool3d(kernel=2) "
            "and MaxPool3d(kernel=3)"
        )

    D2 = (D - 6) // 6 + 1
    H2 = (H - 6) // 6 + 1
    W2 = (W - 6) // 6 + 1

    out = torch.empty((N, C, D2, H2, W2), device=x.device, dtype=x.dtype)
    sN, sC, sD, sH, sW = x.stride()
    oN, oC, oD, oH, oW = out.stride()

    def grid(meta):
        return (
            triton.cdiv(W2, meta['BLOCK_W']),
            triton.cdiv(H2, meta['BLOCK_H']),
            N * C * triton.cdiv(D2, meta['BLOCK_D']),
        )

    _maxpool_6x_3d_kernel[grid](
        x, out,
        N, C, D, H, W,
        sN, sC, sD, sH, sW,
        oN, oC, oD, oH, oW,
        D2, H2, W2,
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError(
                "ModelNew expects Ascend NPU inputs; the Triton kernel path is the only supported runtime."
            )
        if not _is_npu_tensor(self.conv_transpose.weight):
            raise RuntimeError("ModelNew weights must be moved to Ascend NPU before execution.")
        if self.conv_transpose.bias is not None and not _is_npu_tensor(self.conv_transpose.bias):
            raise RuntimeError("ModelNew bias must be moved to Ascend NPU before execution.")

        x = self.conv_transpose(x)
        x = _fused_two_pools_into_one(x)
        return x.sum(dim=1, keepdim=True)
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
