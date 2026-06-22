import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 3
DEFAULT_OUT_CHANNELS = 16
DEFAULT_DEPTH = 16
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_SCALING_FACTOR = 2
DEFAULT_BIAS_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1, 1)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


@triton.jit
def _fused_pointwise_ncdhw_kernel(
    x_ptr,  # *f32
    sf_ptr,  # *f32, shape [C]
    bias_ptr,  # *f32, shape [C]
    out_ptr,  # *f32
    n_elements,  # int
    C,  # int
    DHW,  # int = D*H*W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    m = offs < n_elements

    # Hints for better vectorization/coalescing
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)

    # Load input
    x = tl.load(x_ptr + offs, mask=m, other=0.0)

    c_idx = (offs // DHW) % C
    sf = tl.load(sf_ptr + c_idx, mask=m, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=m, other=0.0)

    # Fused pointwise:
    # 1) scale
    x = x * sf
    # 2) tanh(x) via 2*sigmoid(2x) - 1 (one exp)
    sig2x = 1.0 / (1.0 + tl.exp(-2.0 * x))
    x = 2.0 * sig2x - 1.0
    # 3) multiply by bias
    x = x * b
    # 4) sigmoid
    x = 1.0 / (1.0 + tl.exp(-x))

    # Store
    tl.store(out_ptr + offs, x, mask=m)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        scaling_factor=DEFAULT_SCALING_FACTOR,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor_value = scaling_factor
        self.scaling_factor = nn.Parameter(
            torch.full(bias_shape, float(scaling_factor)))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in {torch.float32, torch.bfloat16}:
            raise RuntimeError(
                f"ModelNew supports only float32 and bfloat16 inputs, got {x.dtype}"
            )

        x = self.conv(x)
        x = x.contiguous()
        _, C, D, H, W = x.shape
        n_elements = x.numel()
        dhw = D * H * W
        sf = self.scaling_factor.reshape(C).contiguous()
        bs = self.bias.reshape(C).contiguous()

        block = 4096

        def grid(META):
            return (triton.cdiv(n_elements, META["BLOCK_SIZE"]), )

        _fused_pointwise_ncdhw_kernel[grid](
            x,
            sf,
            bs,
            x,
            n_elements,
            C,
            dhw,
            BLOCK_SIZE=block,
            num_warps=8,
        )
        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]
