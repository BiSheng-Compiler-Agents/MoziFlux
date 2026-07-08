import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_IN_CHANNELS = 32
DEFAULT_OUT_CHANNELS = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 1
_MAX_BLOCK_C = 1024


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _lse_relu_lastdim_kernel(
    x_ptr,  # contiguous NHWDC-like buffer, flattened [M, C]
    out_ptr,  # flattened output [M]
    M,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < M
    col_mask = cols < C

    offsets = rows[:, None] * C + cols[None, :]
    vals = tl.load(x_ptr + offsets,
                   mask=row_mask[:, None] & col_mask[None, :],
                   other=-float("inf"))
    vals = vals.to(tl.float32)

    m = tl.max(vals, axis=1)
    shifted = vals - m[:, None]
    expv = tl.exp(shifted)
    s = tl.sum(expv, axis=1)
    out = tl.log(s) + m
    out = tl.maximum(out, 0.0)
    tl.store(out_ptr + rows, out, mask=row_mask)


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _lse_relu_triton_lastdim(x_nhwc: torch.Tensor, out_shape) -> torch.Tensor:
    # x_nhwc is contiguous with channel as innermost dimension: [M, C]
    assert x_nhwc.is_contiguous()
    M = x_nhwc.numel() // x_nhwc.shape[-1]
    C = x_nhwc.shape[-1]
    if C > _MAX_BLOCK_C:
        # General fallback for unusually wide channel counts; preserves baseline contract.
        y = torch.logsumexp(x_nhwc.float(), dim=-1)
        y = torch.relu(y)
        return y.reshape(out_shape).to(x_nhwc.dtype)

    block_c = _next_power_of_2(C)
    block_m = 16 if block_c >= 64 else 32
    y_flat = torch.empty((M, ), device=x_nhwc.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), )
    _lse_relu_lastdim_kernel[grid](x_nhwc,
                                   y_flat,
                                   M,
                                   C=C,
                                   BLOCK_M=block_m,
                                   BLOCK_C=block_c)
    return y_flat.reshape(out_shape).to(x_nhwc.dtype)


class ModelNew(nn.Module):
    """Conv3d -> MaxPool3d -> LogSumExp(dim=1, keepdim=True) -> ReLU.

    Optimization: keep Conv3d/MaxPool3d on Ascend ACL, then transpose the pooled tensor so
    channel reduction is contiguous and use a compact Triton row-wise LogSumExp+ReLU kernel.
    """

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise ValueError("ModelNew.forward requires Ascend NPU input")
        x = self.conv(x)
        x = self.max_pool(x)
        n, c, d, h, w = x.shape
        # The baseline reduces over C with stride D*H*W.  Make C contiguous once so the
        # Triton reduction loads [BLOCK_M, BLOCK_C] coalesced rows instead of C strided lanes.
        x_last = x.permute(0, 2, 3, 4, 1).contiguous()
        return _lse_relu_triton_lastdim(x_last, (n, 1, d, h, w))


batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
