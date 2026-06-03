import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


DEFAULT_BATCH_SIZE = 64
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 64
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_KERNEL_SIZE = 3
DEFAULT_MULTIPLIER_SHAPE = (out_channels, 1, 1)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _fused_scale_lrelu_gelu(
    x_ptr,          # *float32, input tensor (NCHW) contiguous
    m_ptr,          # *float32, multiplier tensor flattened with shape (C,)
    y_ptr,          # *float32, output tensor (same shape as x)
    n_elements,     # int32, total elements B*C*H*W
    C,              # int32, number of channels
    HW,             # int32, product H*W
    negative_slope: tl.constexpr,  # float constant
    BLOCK_SIZE: tl.constexpr,      # tile size
):
    pid = tl.program_id(axis=0)
    arange = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + arange
    mask = offsets < n_elements
    tl.multiple_of(offsets, 16)

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")

    # Compute channel index for each element: ((idx // (H*W)) % C)
    plane_idx = offsets // HW
    c_idx = plane_idx % C
    scale = tl.load(m_ptr + c_idx, mask=mask, other=1.0)

    # Compute in fp32 for numerical stability and correctness
    x32 = x.to(tl.float32)
    s32 = scale.to(tl.float32)
    v = x32 * s32

    # Branchless LeakyReLU: v = v + (neg - 1) * min(v, 0)
    v = v + (negative_slope - 1.0) * tl.minimum(v, 0.0)

    # GELU (exact): 0.5 * v * (1 + erf(v / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    e = tl.erf(v * inv_sqrt2)
    y32 = 0.5 * v * (1.0 + e)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    Fused Triton kernel is used to apply: y = GELU(LeakyReLU(conv(x) * multiplier))
    """
    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        multiplier_shape=DEFAULT_MULTIPLIER_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")

        x = self.conv(x)
        x = x.contiguous()
        _, C, H, W = x.shape
        if self.multiplier.numel() != C:
            raise RuntimeError(
                f"Multiplier must contain exactly one value per channel, got {self.multiplier.numel()} for C={C}"
            )

        out = torch.empty_like(x)
        m = self.multiplier.contiguous().reshape(-1).to(device=x.device, dtype=x.dtype)
        n_elements = x.numel()
        if n_elements >= 8192 and (n_elements % 8192 == 0):
            block_size = 8192
            num_warps = 16
            num_stages = 2
        elif n_elements >= 4096 and (n_elements % 4096 == 0):
            block_size = 4096
            num_warps = 8
            num_stages = 3
        elif n_elements >= 2048 and (n_elements % 2048 == 0):
            block_size = 2048
            num_warps = 8
            num_stages = 2
        elif n_elements >= 1024:
            block_size = 1024
            num_warps = 4
            num_stages = 2
        else:
            block_size = 512
            num_warps = 4
            num_stages = 2

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _fused_scale_lrelu_gelu[grid](
            x,
            m,
            out,
            n_elements,
            C,
            H * W,
            self.leaky_relu.negative_slope,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
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
batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='npu')]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]