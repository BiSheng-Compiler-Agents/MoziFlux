import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 16
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 128
DEFAULT_DEPTH = 8
DEFAULT_HEIGHT = 16
DEFAULT_WIDTH = 16
DEFAULT_KERNEL_SIZE = 3
DEFAULT_GROUPS = 8
DEFAULT_BIAS = False
DEFAULT_EPS = 1e-5

_MAX_GRID = 65535
_RELU_BLOCK = 4096
_USE_ACL_DISPATCH = True


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _relu_direct_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _relu_persistent_kernel(x_ptr, y_ptr, n_elements, n_programs,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.maximum(x, 0.0)
        tl.store(y_ptr + offs, y, mask=mask)


def _relu_triton(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n_elements = x.numel()
    n_tiles = triton.cdiv(n_elements, _RELU_BLOCK)
    if n_tiles == 0:
        return y
    if n_tiles > _MAX_GRID:
        n_programs = _MAX_GRID
        _relu_persistent_kernel[(n_programs, )](x,
                                                y,
                                                n_elements,
                                                n_programs,
                                                BLOCK_SIZE=_RELU_BLOCK,
                                                num_warps=8,
                                                num_stages=2)
    else:
        _relu_direct_kernel[(n_tiles, )](x,
                                         y,
                                         n_elements,
                                         BLOCK_SIZE=_RELU_BLOCK,
                                         num_warps=8,
                                         num_stages=2)
    return y


class ModelNew(nn.Module):
    """ConvTranspose3d -> ReLU -> GroupNorm optimized for Ascend NPU."""

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        groups=DEFAULT_GROUPS,
        bias=DEFAULT_BIAS,
        eps=DEFAULT_EPS,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups,
                                       num_channels=out_channels,
                                       eps=eps)

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")
        y = self.conv_transpose(x)
        if _USE_ACL_DISPATCH:
            y = torch.relu(y)
        else:
            if not y.is_contiguous():
                y = y.contiguous()
            y = _relu_triton(y)
        return F.group_norm(
            y,
            self.group_norm.num_groups,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.eps,
        )


_MODEL_CACHE: dict[tuple[str, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (str(x.device), x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    return model(x)


batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W, device='npu')]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]
