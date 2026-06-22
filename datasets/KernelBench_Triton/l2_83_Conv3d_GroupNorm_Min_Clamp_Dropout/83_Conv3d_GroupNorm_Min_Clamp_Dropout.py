import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError:
    torch_npu = None

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 3
DEFAULT_OUT_CHANNELS = 16
DEFAULT_DEPTH = 16
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_GROUPS = 8
DEFAULT_MIN_VALUE = 0.0
DEFAULT_MAX_VALUE = 1.0
DEFAULT_DROPOUT_P = 0.2

_MODEL_CACHE = {}


@triton.jit
def _fill_const_kernel(out_ptr, value, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, value, mask=mask)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


def _conv3d_output_shape(x: torch.Tensor, conv: nn.Conv3d):
    n, _, depth, height, width = x.shape
    kd, kh, kw = conv.kernel_size
    sd, sh, sw = conv.stride
    pd, ph, pw = conv.padding
    dd, dh, dw = conv.dilation
    out_depth = (depth + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    out_height = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_width = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return n, conv.out_channels, out_depth, out_height, out_width


class ModelNew(nn.Module):
    """
    Model that performs Conv3d -> GroupNorm -> minimum -> clamp -> dropout.
    Since minimum(x, min_value) followed by clamp(min=min_value, max=max_value)
    collapses to the constant min_value when max_value >= min_value, the fused
    implementation materializes that constant directly with Triton and only keeps
    dropout as the remaining dynamic op in training mode.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        groups: int = DEFAULT_GROUPS,
        min_value: float = DEFAULT_MIN_VALUE,
        max_value: float = DEFAULT_MAX_VALUE,
        dropout_p: float = DEFAULT_DROPOUT_P,
    ):
        super().__init__()
        if max_value < min_value:
            raise ValueError("ModelNew requires max_value >= min_value")
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects an Ascend NPU tensor input")

        out_shape = _conv3d_output_shape(x, self.conv)
        y = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        if y.numel() == 0:
            return y

        block_size = 1024
        grid = (triton.cdiv(y.numel(), block_size), )
        _fill_const_kernel[grid](
            y,
            self.min_value,
            y.numel(),
            BLOCK_SIZE=block_size,
            num_warps=4,
            num_stages=2,
        )

        if self.training and self.dropout.p > 0.0 and self.min_value != 0.0:
            return self.dropout(y)
        return y


def _set_deterministic_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def conv3d_groupnorm_min_clamp_dropout(x: torch.Tensor) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "conv3d_groupnorm_min_clamp_dropout expects an Ascend NPU tensor")

    key = (str(x.device), x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        _set_deterministic_seed(0)
        model = ModelNew(*get_init_inputs()).eval()
        _MODEL_CACHE[key] = model

    with torch.no_grad():
        return model(x)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
groups = 8
min_value = 0.0
max_value = 1.0
dropout_p = 0.2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, groups, min_value, max_value,
        dropout_p
    ]
