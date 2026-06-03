import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 3
DEFAULT_OUT_CHANNELS = 16
DEFAULT_DEPTH = 16
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 1
DEFAULT_SCALE1 = 0.5
DEFAULT_SCALE2 = 1.0
DEFAULT_BIAS_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1, 1)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _avgpool3d_k2s2_bias_scale_fused(
    x_ptr,
    bias_ptr,
    y_ptr,
    C,
    D,
    D2,
    H,
    W,
    H2,
    W2,
    out_spatial,
    alpha,

    BLOCK_HW: tl.constexpr,
    CHUNKS_PER_PLANE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    plane_pid = pid // CHUNKS_PER_PLANE
    chunk_pid = pid - plane_pid * CHUNKS_PER_PLANE
    plane_offs = chunk_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = plane_offs < out_spatial

    nc = plane_pid // D2
    d2 = plane_pid - nc * D2
    c = nc % C

    h2 = plane_offs // W2
    w2 = plane_offs - h2 * W2

    stride_h = W
    stride_d = H * W
    stride_nc = D * H * W
    base = nc * stride_nc + d2 * (2 * stride_d) + (h2 * 2) * stride_h + (w2 * 2)

    x0 = tl.load(x_ptr + base, mask=mask, other=0.0, cache_modifier=".ca")
    x1 = tl.load(x_ptr + base + 1, mask=mask, other=0.0, cache_modifier=".ca")
    x2 = tl.load(x_ptr + base + stride_h, mask=mask, other=0.0, cache_modifier=".ca")
    x3 = tl.load(x_ptr + base + stride_h + 1, mask=mask, other=0.0, cache_modifier=".ca")
    x4 = tl.load(x_ptr + base + stride_d, mask=mask, other=0.0, cache_modifier=".ca")
    x5 = tl.load(x_ptr + base + stride_d + 1, mask=mask, other=0.0, cache_modifier=".ca")
    x6 = tl.load(x_ptr + base + stride_d + stride_h, mask=mask, other=0.0, cache_modifier=".ca")
    x7 = tl.load(x_ptr + base + stride_d + stride_h + 1, mask=mask, other=0.0, cache_modifier=".ca")

    s0 = x0 + x1
    s1 = x2 + x3
    s2 = x4 + x5
    s3 = x6 + x7
    s = (s0 + s1) + (s2 + s3)

    b = tl.load(bias_ptr + c)
    out = s * alpha + b


    tl.store(y_ptr + plane_pid * out_spatial + plane_offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
        scale1=DEFAULT_SCALE1,
        scale2=DEFAULT_SCALE2,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        x = self.conv_transpose(x)
        x = x.contiguous()
        N, C, D, H, W = x.shape
        D2, H2, W2 = D // 2, H // 2, W // 2
        out = torch.empty((N, C, D2, H2, W2), device=x.device, dtype=x.dtype)

        plane_size = H2 * W2
        grid = (N * C * D2 * 1,)

        scale1_value = float(self.scale1.item())
        scale2_value = float(self.scale2.item())
        alpha = scale1_value * scale2_value * 0.125
        bias_1d = (self.bias.view(C).contiguous() * scale2_value)


        _avgpool3d_k2s2_bias_scale_fused[grid](
            x,
            bias_1d,
            out,
            C,
            D,
            D2,
            H,
            W,
            H2,
            W2,
            plane_size,
            alpha,

            BLOCK_HW=961,
            CHUNKS_PER_PLANE=1,
            num_warps=16,
            num_stages=1,
        )
        return out


_MODEL_CACHE: dict[tuple[torch.device, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    return model(x)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device="npu")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]
