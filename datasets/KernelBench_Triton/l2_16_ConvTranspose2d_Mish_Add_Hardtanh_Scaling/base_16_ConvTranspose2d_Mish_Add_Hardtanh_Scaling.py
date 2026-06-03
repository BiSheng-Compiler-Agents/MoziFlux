import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 64
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 2
DEFAULT_PADDING = 1
DEFAULT_OUTPUT_PADDING = 1
DEFAULT_ADD_VALUE = 0.5
DEFAULT_SCALE = 2


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _fused_mish_add_hardtanh_scale_kernel(
    x_ptr, y_ptr,
    n_elements,
    add_value, scale_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hints for better vectorization/coalescing
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    mask = offs < n_elements

    # Load; values are single-use so prefer evict_last
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last")
    x_f32 = x.to(tl.float32)

    # Mish: x * tanh(softplus(x))
    # Stable softplus: max(x, 0) + log(1 + exp(-abs(x)))
    ax = tl.abs(x_f32)
    sp = tl.maximum(x_f32, 0.0) + tl.log(1.0 + tl.exp(-ax))

    tanh_sp = tl.tanh(sp)
    mish = x_f32 * tanh_sp

    # Add, clamp to [-1, 1] (hardtanh), then scale
    out = mish + add_value
    out = tl.clamp(out, -1.0, 1.0)
    out = (out * scale_value).to(x.dtype)

    tl.store(y_ptr + offs, out, mask=mask)


def _fused_mish_add_hardtanh_scale(x: torch.Tensor, add_value: float, scale: float) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError("_fused_mish_add_hardtanh_scale expects input tensors on Ascend NPU")
    if x.numel() == 0:
        return x.clone()

    # Work in-place to reduce memory traffic and allocations
    x_contig = x.contiguous()
    n_elements = x_contig.numel()
    BLOCK_SIZE = 4032
    max_programs = 65535
    chunk_elems = BLOCK_SIZE * max_programs
    for base in range(0, n_elements, chunk_elems):
        current_elems = min(chunk_elems, n_elements - base)
        grid = (triton.cdiv(current_elems, BLOCK_SIZE),)
        # Alias output to input for in-place epilogue and keep each launch within the grid cap.
        _fused_mish_add_hardtanh_scale_kernel[grid](
            x_contig.view(-1)[base:], x_contig.view(-1)[base:],
            current_elems,
            float(add_value), float(scale),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=3,
        )
    return x_contig


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies Mish activation, adds a value, 
    applies Hardtanh activation, and scales the output.
    """
    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
        output_padding=DEFAULT_OUTPUT_PADDING,
        add_value=DEFAULT_ADD_VALUE,
        scale=DEFAULT_SCALE,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        x = self.conv_transpose(x)
        # Fused Triton kernel: mish -> add -> hardtanh -> scale
        x = _fused_mish_add_hardtanh_scale(x, self.add_value, self.scale)
        return x


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
in_channels  = 64  
out_channels = 64  
height = width = 128  
kernel_size  = 3
stride       = 2  
padding      = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='npu')]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]
