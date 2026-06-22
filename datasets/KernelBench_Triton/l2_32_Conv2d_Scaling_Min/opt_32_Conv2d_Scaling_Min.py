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


_MAX_PROGRAMS = 65535
_BLOCK_HW = 256
_BLOCK_C = 16

if HAS_TRITON:

    @triton.jit
    def _scale_min_channel_direct_kernel(
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
        BLOCK_HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_hw_tiles = tl.cdiv(H * W, BLOCK_HW)
        b_idx = pid // n_hw_tiles
        hw_tile = pid - b_idx * n_hw_tiles
        offs_hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask_hw = (b_idx < B) & (offs_hw < H * W)
        h = offs_hw // W
        w = offs_hw - h * W
        offs_c = tl.arange(0, BLOCK_C)
        acc = tl.full((BLOCK_HW, ), float("inf"), tl.float32)
        for c0 in range(0, C, BLOCK_C):
            c_idx = c0 + offs_c
            mask = (c_idx[:, None] < C) & mask_hw[None, :]
            vals = tl.load(
                x_ptr + b_idx * stride_xn + c_idx[:, None] * stride_xc +
                h[None, :] * stride_xh + w[None, :] * stride_xw,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            vals = tl.where(mask, vals * scale, float("inf"))
            acc = tl.minimum(acc, tl.min(vals, axis=0).to(tl.float32))
        tl.store(
            y_ptr + b_idx * stride_yn + h * stride_yh + w * stride_yw,
            acc.to(y_ptr.dtype.element_ty),
            mask=mask_hw,
        )

    @triton.jit
    def _scale_min_channel_persistent_kernel(
        x_ptr,
        y_ptr,
        scale,
        B,
        H,
        W,
        n_tiles,
        n_programs,
        stride_xn,
        stride_xc,
        stride_xh,
        stride_xw,
        stride_yn,
        stride_yc,
        stride_yh,
        stride_yw,
        C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_hw_tiles = tl.cdiv(H * W, BLOCK_HW)
        for tile_id in range(pid, n_tiles, n_programs):
            b_idx = tile_id // n_hw_tiles
            hw_tile = tile_id - b_idx * n_hw_tiles
            offs_hw = (hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)).to(
                tl.int64)
            mask_hw = (b_idx < B) & (offs_hw < H * W)
            h = offs_hw // W
            w = offs_hw - h * W
            offs_c = tl.arange(0, BLOCK_C)
            acc = tl.full((BLOCK_HW, ), float("inf"), tl.float32)
            for c0 in range(0, C, BLOCK_C):
                c_idx = c0 + offs_c
                mask = (c_idx[:, None] < C) & mask_hw[None, :]
                vals = tl.load(
                    x_ptr + b_idx * stride_xn + c_idx[:, None] * stride_xc +
                    h[None, :] * stride_xh + w[None, :] * stride_xw,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                vals = tl.where(mask, vals * scale, float("inf"))
                acc = tl.minimum(acc, tl.min(vals, axis=0).to(tl.float32))
            tl.store(
                y_ptr + b_idx * stride_yn + h * stride_yh + w * stride_yw,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_hw,
            )
else:

    def _scale_min_channel_direct_kernel(*args, **kwargs):
        raise RuntimeError("Triton is unavailable")

    def _scale_min_channel_persistent_kernel(*args, **kwargs):
        raise RuntimeError("Triton is unavailable")


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0


class ModelNew(nn.Module):
    """Conv2d followed by scale then channelwise minimum."""

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=None,
                 scale_factor=None,
                 force_triton: bool = False):
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
        self.force_triton = bool(force_triton)

    def _triton_epilogue(self, x):
        B, C, H, W = x.shape
        y = torch.empty((B, 1, H, W), device=x.device, dtype=x.dtype)
        n_tiles = triton.cdiv(B * H * W, _BLOCK_HW)
        # tile count is B * ceil(HW/BLOCK_HW), not ceil(B*H*W/BLOCK_HW)
        n_tiles = B * triton.cdiv(H * W, _BLOCK_HW)
        if n_tiles > _MAX_PROGRAMS:
            n_programs = _MAX_PROGRAMS
            _scale_min_channel_persistent_kernel[(n_programs, )](
                x,
                y,
                self.scale_factor,
                B,
                H,
                W,
                n_tiles,
                n_programs,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                x.stride(3),
                y.stride(0),
                y.stride(1),
                y.stride(2),
                y.stride(3),
                C=C,
                BLOCK_HW=_BLOCK_HW,
                BLOCK_C=_BLOCK_C,
                num_warps=4,
                num_stages=2,
            )
        else:
            _scale_min_channel_direct_kernel[(n_tiles, )](
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
                BLOCK_HW=_BLOCK_HW,
                BLOCK_C=_BLOCK_C,
                num_warps=4,
                num_stages=2,
            )
        return y

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise ValueError(
                "ModelNew.forward expects an Ascend NPU tensor input.")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "ModelNew.forward only supports inference under torch.no_grad()."
            )
        if not _is_npu_tensor(self.conv.weight):
            raise ValueError(
                "Model parameters must be moved to NPU before execution.")
        if self.conv.bias is not None and not _is_npu_tensor(self.conv.bias):
            raise ValueError(
                "Model bias must be moved to NPU before execution.")

        x = F.conv2d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        if self.force_triton:
            if not HAS_TRITON:
                raise RuntimeError("Triton is required for force_triton=True")
            return self._triton_epilogue(x)
        # scale is a constructor scalar. For non-negative scale, min(scale*x)=scale*min(x);
        # for negative scale, min(scale*x)=scale*max(x). This removes the custom Triton
        # epilogue and one full-output multiply before the channel reduction.
        if self.scale_factor >= 0.0:
            return torch.amin(x, dim=1, keepdim=True).mul(self.scale_factor)
        return torch.amax(x, dim=1, keepdim=True).mul(self.scale_factor)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
