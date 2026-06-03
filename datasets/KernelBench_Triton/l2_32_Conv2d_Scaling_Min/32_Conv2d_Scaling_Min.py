import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


if HAS_TRITON:
    @triton.jit
    def _scale_min_channel_kernel(
        x_ptr,
        y_ptr,
        scale,
        B,
        H,
        W,
        stride_xn,
        stride_xc,
        stride_xh,
        stride_xw,
        stride_yn,
        stride_yc,
        stride_yh,
        stride_yw,
        C: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_hw = tl.program_id(axis=0)
        pid_b = tl.program_id(axis=1)
        b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        hw_start = pid_hw * BLOCK_HW
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_b = b_idx < B
        mask_hw = offs_hw < (H * W)
        h = offs_hw // W
        w = offs_hw % W

        x_base = (
            b_idx[:, None] * stride_xn
            + h[None, :] * stride_xh
            + w[None, :] * stride_xw
        )
        offs_c = tl.arange(0, BLOCK_C)
        acc = tl.full((BLOCK_B, BLOCK_HW), float("inf"), tl.float32)

        for c0 in range(0, C, BLOCK_C):
            c_idx = c0 + offs_c[None, None, :]
            mask_c = c_idx < C
            mask = mask_b[:, None, None] & mask_hw[None, :, None] & mask_c
            x_vals = tl.load(
                x_ptr + x_base[:, :, None] + c_idx * stride_xc,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            scaled_vals = tl.where(mask, x_vals * scale, float("inf"))
            block_min = tl.min(scaled_vals, axis=2).to(tl.float32)
            acc = tl.minimum(acc, block_min)

        y_offs = (
            b_idx[:, None] * stride_yn
            + h[None, :] * stride_yh
            + w[None, :] * stride_yw
        )
        tl.store(
            y_ptr + y_offs,
            acc.to(y_ptr.dtype.element_ty),
            mask=mask_b[:, None] & mask_hw[None, :],
        )
else:
    def _scale_min_channel_kernel(*args, **kwargs):
        raise RuntimeError("Triton is unavailable, so _scale_min_channel_kernel cannot run.")


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scale_factor = 2.0


class ModelNew(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        kernel_size=None,
        scale_factor=None,
    ):
        super(ModelNew, self).__init__()
        if in_channels is None:
            in_channels = globals()["in_channels"]
        if out_channels is None:
            out_channels = globals()["out_channels"]
        if kernel_size is None:
            kernel_size = globals()["kernel_size"]
        if scale_factor is None:
            scale_factor = globals()["scale_factor"]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, out_height, out_width).
        """
        if not HAS_TRITON:
            raise RuntimeError("Triton is required for ModelNew.forward.")
        if not _is_npu_tensor(x):
            raise ValueError("ModelNew.forward expects an Ascend NPU tensor input.")
        if torch.is_grad_enabled():
            raise RuntimeError("ModelNew.forward only supports inference under torch.no_grad().")

        if not _is_npu_tensor(self.conv.weight):
            raise ValueError("Model parameters must be moved to NPU before execution.")
        if self.conv.bias is not None and not _is_npu_tensor(self.conv.bias):
            raise ValueError("Model bias must be moved to NPU before execution.")

        x = F.conv2d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

        if x.dim() != 4 or x.shape[1] <= 0:
            raise ValueError("Conv2d output must be a non-empty 4D tensor.")

        B, C, H, W = x.shape
        y = torch.empty((B, 1, H, W), device=x.device, dtype=x.dtype)

        def _next_pow2(v: int) -> int:
            return 1 if v <= 1 else 1 << int(math.ceil(math.log2(v)))

        BLOCK_B = 8
        BLOCK_HW = 8
        BLOCK_C = min(32, max(16, _next_pow2(C)))
        grid = (triton.cdiv(H * W, BLOCK_HW), triton.cdiv(B, BLOCK_B))
        _scale_min_channel_kernel[grid](
            x,
            y,
            self.scale_factor,
            B,
            H,
            W,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            y.stride(3),
            C=C,
            BLOCK_B=BLOCK_B,
            BLOCK_HW=BLOCK_HW,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )
        return y
batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
