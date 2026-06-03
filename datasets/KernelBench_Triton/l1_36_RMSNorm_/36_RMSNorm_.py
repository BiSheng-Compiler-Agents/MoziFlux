import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_nchw_kernel(
    x_ptr,
    y_ptr,
    rows,
    cols,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    eps,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    row_x_ptr = x_ptr + pid * stride_xm
    row_y_ptr = y_ptr + pid * stride_ym
    offsets = tl.arange(0, BLOCK_C)
    col_offs_x = offsets * stride_xn
    col_offs_y = offsets * stride_yn

    sumsq = tl.zeros([1], dtype=tl.float32)
    c = 0
    while c < cols:
        col_ids = c + offsets
        mask = col_ids < cols
        x = tl.load(row_x_ptr + c * stride_xn + col_offs_x, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sumsq += tl.sum(x_f32 * x_f32, axis=0)
        c += BLOCK_C

    inv_rms = tl.rsqrt(sumsq / cols + eps)

    c = 0
    while c < cols:
        col_ids = c + offsets
        mask = col_ids < cols
        x = tl.load(row_x_ptr + c * stride_xn + col_offs_x, mask=mask, other=0.0)
        y = x * inv_rms
        tl.store(row_y_ptr + c * stride_yn + col_offs_y, y, mask=mask)
        c += BLOCK_C


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.device.type != "npu":
        raise ValueError("rms_norm expects an Ascend NPU tensor")
    if x.dim() != 4:
        raise ValueError(f"rms_norm expects a 4D NCHW tensor, got shape {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("rms_norm expects a contiguous tensor")

    B, C, H, W = x.shape
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_2d = x_nhwc.view(B * H * W, C)
    y_2d = torch.empty_like(x_2d)
    stride_xm, stride_xn = x_2d.stride()
    stride_ym, stride_yn = y_2d.stride()
    if C >= 4096:
        block_c = 4096
    elif C >= 2048:
        block_c = 2048
    elif C >= 1024:
        block_c = 1024
    elif C >= 512:
        block_c = 512
    elif C >= 256:
        block_c = 256
    else:
        block_c = 128

    _rmsnorm_nchw_kernel[(x_2d.shape[0],)](
        x_2d,
        y_2d,
        x_2d.shape[0],
        C,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        float(eps),
        BLOCK_C=block_c,
    )
    return y_2d.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class ModelNew(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        if x.size(1) != self.num_features:
            raise ValueError(
                f"expected channel dimension {self.num_features}, got {x.size(1)}"
            )
        return rms_norm(x, self.eps)
batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]
def get_init_inputs():
    return [features]