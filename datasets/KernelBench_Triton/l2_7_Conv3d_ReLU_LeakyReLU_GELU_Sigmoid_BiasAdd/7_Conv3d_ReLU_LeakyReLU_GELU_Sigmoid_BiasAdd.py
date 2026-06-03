import torch
import torch.nn as nn
import triton
import triton.language as tl


DEFAULT_BATCH_SIZE = 64
DEFAULT_IN_CHANNELS = 8
DEFAULT_OUT_CHANNELS = 32
DEFAULT_DEPTH = 32
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_BIAS_SHAPE = (out_channels, 1, 1, 1)

_MODEL_CACHE = {}


@triton.jit
def _fused_post_ops_bias_kernel(
    x_ptr,             # *f32
    bias_ptr,          # *f32
    y_ptr,             # *f32
    n_elements,        # i32
    C,                 # i32
    stride_c,          # i32 (elements)
    bias_stride_c,     # i32 (elements)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Hints to compiler for better vectorization/coalescing
    tl.multiple_of(block_start, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)

    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # 1) ReLU
    x = tl.maximum(x, 0.0)

    # 2) LeakyReLU after ReLU is a no-op; omit to save work while preserving semantics

    # 3) GELU (exact): 0.5 * u * (1 + erf(u / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    u = x
    e = tl.math.erf(u * inv_sqrt2)
    x = (u * (1.0 + e)) * 0.5

    # 4) Sigmoid: since x >= 0 after ReLU->GELU, use simplified stable form
    # Use exp2 for slightly faster evaluation: exp(-x) = exp2(-x * log2(e))
    LOG2E = 1.4426950408889634
    x = 1.0 / (1.0 + tl.exp2(-x * LOG2E))

    # Map each flattened position back to its channel for broadcast bias add.
    c_idx = ((offs // stride_c) % C).to(tl.int32)
    b = tl.load(bias_ptr + c_idx * bias_stride_c, mask=mask, other=0.0)
    out = x + b

    tl.store(y_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    Fused the post-conv elementwise ops + bias addition into a single Triton kernel for performance.
    """
    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects an Ascend NPU tensor input")

        if self.conv.weight.device != x.device or self.conv.weight.dtype != x.dtype:
            self.conv = self.conv.to(device=x.device, dtype=x.dtype)
        if self.bias.device != x.device or self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(device=x.device, dtype=x.dtype)

        y = self.conv(x).contiguous()
        n, c, d, h, w = y.shape
        n_elements = y.numel()
        if n_elements == 0:
            return y

        b = self.bias.contiguous()
        stride_c = y.stride(1)
        bias_stride_c = b.stride(0)
        block = 1024
        grid = lambda meta: (triton.cdiv(n_elements, block),)
        _fused_post_ops_bias_kernel[grid](
            y,
            b,
            y,
            n_elements,
            c,
            stride_c,
            bias_stride_c,
            BLOCK_SIZE=block,
            num_warps=4,
            num_stages=2,
        )
        return y


def _set_deterministic_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def conv3d_relu_leakyrelu_gelu_sigmoid_biasadd(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError(
            "conv3d_relu_leakyrelu_gelu_sigmoid_biasadd expects an Ascend NPU tensor"
        )

    key = (str(x.device), x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        _set_deterministic_seed(0)
        model = ModelNew(*get_init_inputs()).eval().to(device=x.device, dtype=x.dtype)
        _MODEL_CACHE[key] = model

    with torch.no_grad():
        return model(x)
batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]