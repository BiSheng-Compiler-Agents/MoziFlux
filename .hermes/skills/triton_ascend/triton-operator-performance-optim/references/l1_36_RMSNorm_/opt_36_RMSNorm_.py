import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_nchw_hw_kernel(
    x_ptr,
    y_ptr,
    hw,
    channels,
    stride_xb,
    stride_xc,
    stride_yb,
    stride_yc,
    eps,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < hw
    offs_c = tl.arange(0, BLOCK_C)

    sumsq = tl.zeros([BLOCK_HW], dtype=tl.float32)
    c = 0
    while c < channels:
        c_ids = c + offs_c
        mask_c = c_ids < channels
        x_ptrs = x_ptr + pid_b * stride_xb + c_ids[:,
                                                   None] * stride_xc + offs_hw[
                                                       None, :]
        x = tl.load(x_ptrs, mask=mask_c[:, None] & mask_hw[None, :], other=0.0)
        x_f32 = x.to(tl.float32)
        sumsq += tl.sum(x_f32 * x_f32, axis=0)
        c += BLOCK_C

    inv_rms = tl.rsqrt(sumsq / channels + eps)

    c = 0
    while c < channels:
        c_ids = c + offs_c
        mask_c = c_ids < channels
        x_ptrs = x_ptr + pid_b * stride_xb + c_ids[:,
                                                   None] * stride_xc + offs_hw[
                                                       None, :]
        y_ptrs = y_ptr + pid_b * stride_yb + c_ids[:,
                                                   None] * stride_yc + offs_hw[
                                                       None, :]
        x = tl.load(x_ptrs, mask=mask_c[:, None] & mask_hw[None, :], other=0.0)
        y = x * inv_rms[None, :]
        tl.store(y_ptrs, y, mask=mask_c[:, None] & mask_hw[None, :])
        c += BLOCK_C


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.device.type != "npu":
        raise ValueError("rms_norm expects an Ascend NPU tensor")
    if x.dim() != 4:
        raise ValueError(
            f"rms_norm expects a 4D NCHW tensor, got shape {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("rms_norm expects a contiguous tensor")

    B, C, H, W = x.shape
    hw = H * W
    x_3d = x.view(B, C, hw)
    y_3d = torch.empty_like(x_3d)
    stride_xb, stride_xc, _ = x_3d.stride()
    stride_yb, stride_yc, _ = y_3d.stride()

    if C <= 64:
        block_c = 64
    elif C <= 128:
        block_c = 128
    elif C <= 256:
        block_c = 256
    else:
        block_c = 128

    block_hw = 128

    _rmsnorm_nchw_hw_kernel[(triton.cdiv(hw, block_hw), B)](
        x_3d,
        y_3d,
        hw,
        C,
        stride_xb,
        stride_xc,
        stride_yb,
        stride_yc,
        float(eps),
        BLOCK_HW=block_hw,
        BLOCK_C=block_c,
    )
    return y_3d.view(B, C, H, W)


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
