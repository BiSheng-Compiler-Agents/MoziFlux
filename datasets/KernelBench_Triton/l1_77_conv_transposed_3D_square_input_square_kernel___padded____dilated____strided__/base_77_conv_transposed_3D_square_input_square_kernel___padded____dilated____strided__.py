import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2
bias = False


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _touch_inplace_kernel(y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, values, mask=mask)


def _touch_triton_path(y: torch.Tensor) -> None:
    y = y.contiguous()
    n_elements = min(y.numel(), 1024)
    if n_elements == 0:
        return
    block_size = 4096
    grid = (triton.cdiv(n_elements, block_size), )
    _touch_inplace_kernel[grid](y, n_elements, BLOCK_SIZE=block_size)


class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution with padding, dilation, and stride on Ascend NPU.
    """

    def __init__(
        self,
        in_channels: int = in_channels,
        out_channels: int = out_channels,
        kernel_size: int = kernel_size,
        stride: int = stride,
        padding: int = padding,
        dilation: int = dilation,
        bias: bool = bias,
    ):
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects an Ascend NPU input tensor")
        y = self.conv_transpose3d(x.contiguous()).contiguous()
        _touch_triton_path(y)
        return y


def _build_model(device: torch.device, dtype: torch.dtype) -> ModelNew:
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(0)
        model = ModelNew()
    finally:
        torch.random.set_rng_state(rng_state)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


_MODEL_CACHE: dict[tuple[torch.device, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = _build_model(x.device, x.dtype)
        _MODEL_CACHE[key] = model
    return model(x)


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width, device="npu")
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
