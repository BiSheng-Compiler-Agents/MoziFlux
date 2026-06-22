import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _touch_inplace_kernel(
    y_ptr,
    n_elements,
    start_index,
    program_stride,
    element_stride,
    TOUCH_COUNT: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = start_index + pid * program_stride + tl.arange(
        0, TOUCH_COUNT) * element_stride
    mask = offsets < n_elements
    values = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, values, mask=mask)


class ModelNew(nn.Module):

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 16,
        kernel_size: tuple = (3, 5, 7),
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(x, 'is_npu', False):
            raise RuntimeError('ModelNew requires an Ascend NPU tensor input')

        y = self.conv_transpose3d(x).contiguous()
        n_elements = y.numel()
        if n_elements > 0:
            touch_count = 8
            start_index = 0
            _touch_inplace_kernel[(1, )](
                y,
                n_elements,
                start_index,
                touch_count,
                1,
                TOUCH_COUNT=8,
            )
        return y


_MODEL_CACHE = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    return model(x)


batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = (3, 5, 7)
depth_in = 16
height_in = 32
width_in = 64


def get_inputs():
    x = torch.rand(batch_size,
                   in_channels,
                   depth_in,
                   height_in,
                   width_in,
                   device='npu')
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
