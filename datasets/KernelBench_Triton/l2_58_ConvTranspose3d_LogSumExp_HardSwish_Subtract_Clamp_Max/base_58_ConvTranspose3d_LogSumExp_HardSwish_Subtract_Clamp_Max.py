import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 3
DEFAULT_OUT_CHANNELS = 16
DEFAULT_DEPTH = 16
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 1
DEFAULT_BIAS_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1, 1)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


if TRITON_AVAILABLE:
    @triton.jit
    def _lse_hswish_bias_clamp_kernel(
        x_ptr,             # *const float
        out_ptr,           # *float
        min_bias_ptr,      # *const float (1 element)
        M,                 # int32: total number of elements per (N*D*H*W)
        STRIDE_C,          # int32: stride between channels (D*H*W)
        C: tl.constexpr,   # number of channels to reduce over (compile-time constant)
        BLOCK: tl.constexpr,  # block size along M
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < M

        min_bias = tl.load(min_bias_ptr).to(tl.float32)
        neg_inf = -float("inf")
        base_ptr = x_ptr + offs

        # Stream logsumexp in one pass to avoid reloading every channel twice.
        m = tl.full((BLOCK,), neg_inf, dtype=tl.float32)
        s = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in tl.static_range(0, C):
            v = tl.load(base_ptr + c * STRIDE_C, mask=mask, other=neg_inf).to(tl.float32)
            new_m = tl.maximum(m, v)
            s = s * tl.exp(m - new_m) + tl.exp(v - new_m)
            m = new_m

        lse = tl.where(mask, m + tl.log(s), 0.0)
        t = lse + 3.0
        sig = 1.0 / (1.0 + tl.exp(-t))
        h = lse * sig * (1.0 / 6.0)
        z = h - min_bias
        z = tl.maximum(tl.minimum(z, 1.0), -1.0)
        tl.store(out_ptr + offs, z, mask=mask)
else:
    _lse_hswish_bias_clamp_kernel = None


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp, and maximum operations.
    Fused with a Triton kernel:
      After conv_transpose:
        - compute logsumexp over channels,
        - apply HardSwish,
        - subtract min(bias) and clamp,
        - result equals max over channels of clamp(h - bias_c) due to clamp monotonicity.
    """
    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for ModelNew, but it is not available in this environment.")
        if not _is_npu_tensor(x):
            raise RuntimeError(f"ModelNew requires NPU inputs, but received device={x.device}.")

        weight = self.conv_transpose.weight.to(dtype=x.dtype)
        bias = None if self.conv_transpose.bias is None else self.conv_transpose.bias.to(dtype=x.dtype)
        y = F.conv_transpose3d(
            x,
            weight,
            bias=bias,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            groups=self.conv_transpose.groups,
            dilation=self.conv_transpose.dilation,
        )
        if not _is_npu_tensor(y):
            raise RuntimeError(f"ConvTranspose3d output must stay on NPU, but received device={y.device}.")

        B, C, D, H, W = y.shape
        M = B * D * H * W
        y = y.contiguous()
        stride_c = y.stride(1)
        out = torch.empty((B, 1, D, H, W), device=y.device, dtype=y.dtype).contiguous()
        min_bias_t = self.bias.amin().reshape(1).to(device=y.device, dtype=y.dtype).contiguous()

        # Launch Triton kernel
        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK"]),)

        _lse_hswish_bias_clamp_kernel[grid](
            y.view(-1),                # x_ptr
            out.view(-1),              # out_ptr flattened
            min_bias_t,                # min_bias_ptr
            M,                         # total elements over (N*D*H*W)
            stride_c,                  # stride between channels
            C=C,                       # number of channels (constexpr)
            BLOCK=3072,                # auto round block
            num_warps=4,               # auto round warps
            num_stages=3,
        )

        return out

batch_size = DEFAULT_BATCH_SIZE
in_channels = DEFAULT_IN_CHANNELS
out_channels = DEFAULT_OUT_CHANNELS
depth, height, width = DEFAULT_DEPTH, DEFAULT_HEIGHT, DEFAULT_WIDTH
kernel_size = DEFAULT_KERNEL_SIZE
stride = DEFAULT_STRIDE
padding = DEFAULT_PADDING
bias_shape = DEFAULT_BIAS_SHAPE

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
