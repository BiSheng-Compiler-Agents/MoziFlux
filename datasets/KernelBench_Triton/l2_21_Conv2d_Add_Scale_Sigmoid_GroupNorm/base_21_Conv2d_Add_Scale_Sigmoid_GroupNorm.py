import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 8
DEFAULT_OUT_CHANNELS = 32
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_KERNEL_SIZE = 3
DEFAULT_NUM_GROUPS = 8


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=1),
    ],
    key=["full_nhw_elements"],
)
@triton.jit
def _bias_scale_sigmoid_kernel_full(
    x_ptr,
    bias_ptr,
    scale_ptr,
    y_ptr,
    full_nhw_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid_hw = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    offs = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
    base = pid_c * full_nhw_elements + offs
    x = tl.load(x_ptr + base)
    b = tl.load(bias_ptr + pid_c)
    s = tl.load(scale_ptr + pid_c)
    z = (x + b) * s
    y = tl.sigmoid(z)
    tl.store(y_ptr + base, y)


@triton.jit
def _bias_scale_sigmoid_kernel_tail(
    x_ptr,
    bias_ptr,
    scale_ptr,
    y_ptr,
    nhw_elements,
    tail_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    offs = tail_offset + tl.arange(0, BLOCK_SIZE)
    mask = offs < nhw_elements
    base = pid_c * nhw_elements + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    b = tl.load(bias_ptr + pid_c)
    s = tl.load(scale_ptr + pid_c)
    z = (x + b) * s
    y = tl.sigmoid(z)
    tl.store(y_ptr + base, y, mask=mask)


def fused_bias_scale_sigmoid(x: torch.Tensor, bias: torch.Tensor,
                             scale: torch.Tensor):
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "fused_bias_scale_sigmoid expects input tensors on Ascend NPU")
    if x.requires_grad:
        raise RuntimeError(
            "fused_bias_scale_sigmoid does not support autograd-enabled inputs"
        )
    if x.dtype not in (torch.float16, torch.float32):
        raise RuntimeError(
            "fused_bias_scale_sigmoid supports only float16 and float32 inputs"
        )

    x_contig = x.contiguous()
    bias_contig = bias.contiguous().view(-1)
    scale_contig = scale.contiguous().view(-1)

    _, C, H, W = x_contig.shape
    nhw_elements = x_contig.shape[0] * H * W

    block_size = 8192
    full_blocks = nhw_elements // block_size
    full_nhw_elements = full_blocks * block_size

    y = torch.empty_like(x_contig)

    if full_blocks > 0:

        def grid(META):
            return (full_blocks, C)

        _bias_scale_sigmoid_kernel_full[grid](x_contig, bias_contig,
                                              scale_contig, y,
                                              full_nhw_elements)
    if full_nhw_elements < nhw_elements:
        _bias_scale_sigmoid_kernel_tail[(C, )](
            x_contig,
            bias_contig,
            scale_contig,
            y,
            nhw_elements,
            full_nhw_elements,
            BLOCK_SIZE=1024,
        )
    return y


class ModelNew(nn.Module):

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        num_groups=DEFAULT_NUM_GROUPS,
        bias_shape=None,
        scale_shape=None,
    ):
        super(ModelNew, self).__init__()
        if bias_shape is None:
            bias_shape = (out_channels, 1, 1)
        if scale_shape is None:
            scale_shape = (out_channels, 1, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")
        if x.dtype not in (torch.float16, torch.float32):
            raise RuntimeError(
                "ModelNew supports only float16 and float32 inputs")

        x = self.conv(x)
        x = fused_bias_scale_sigmoid(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x


_MODEL_CACHE: dict[tuple[str, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (str(x.device), x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    return model(x)


batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="npu")]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, num_groups, bias_shape,
        scale_shape
    ]
