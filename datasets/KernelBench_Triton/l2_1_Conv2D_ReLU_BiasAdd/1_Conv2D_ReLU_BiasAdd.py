import torch
import torch.nn as nn
import torch_npu  # noqa: F401

import triton
import triton.language as tl


DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 128
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_KERNEL_SIZE = 3
DEFAULT_BIAS_SHAPE = (out_channels, 1, 1)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _relu_add_bias_kernel(
    x_ptr,
    y_ptr,
    b_ptr,
    N,
    C,
    H,
    W,
    BLOCK_W: tl.constexpr,
):
    pid_nch = tl.program_id(axis=0)
    pid_wblk = tl.program_id(axis=1)

    n = pid_nch // (C * H)
    rem = pid_nch % (C * H)
    c = rem // H
    h = rem % H

    start_w = pid_wblk * BLOCK_W
    w_offsets = start_w + tl.arange(0, BLOCK_W)
    mask = w_offsets < W

    base = ((n * C + c) * H + h) * W
    offs = base + w_offsets
    tl.max_contiguous(offs, BLOCK_W)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    b = tl.load(b_ptr + c)
    x = x + b
    tl.store(y_ptr + offs, x, mask=mask)


def _relu_add_bias_triton(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if not _is_npu_tensor(x) or not _is_npu_tensor(bias):
        raise RuntimeError("_relu_add_bias_triton expects NPU tensors")
    if x.requires_grad:
        raise RuntimeError("_relu_add_bias_triton does not support autograd-enabled input tensors")
    if x.dtype not in (torch.float16, torch.float32):
        raise RuntimeError("_relu_add_bias_triton supports only float16 and float32 tensors")
    if x.ndim != 4:
        raise RuntimeError(f"Expected x to have shape [N, C, H, W], got {tuple(x.shape)}")
    if bias.numel() != x.shape[1]:
        raise RuntimeError(
            f"Bias must contain exactly one value per channel, got {bias.numel()} for C={x.shape[1]}"
        )

    x = x.contiguous()
    bias_flat = bias.contiguous().reshape(-1).to(device=x.device, dtype=x.dtype)
    N, C, H, W = x.shape
    y = torch.empty_like(x)

    if W >= 128:
        block_w = 128
    elif W >= 64:
        block_w = 64
    else:
        block_w = 32

    grid = (N * C * H, triton.cdiv(W, block_w))
    _relu_add_bias_kernel[grid](
        x,
        y,
        bias_flat,
        N,
        C,
        H,
        W,
        BLOCK_W=block_w,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Performs Conv2d -> ReLU -> BiasAdd, using Triton for the fused post-conv stage.
    """

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        bias_shape=DEFAULT_BIAS_SHAPE,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")

        x = self.conv(x)
        return _relu_add_bias_triton(x, self.bias)


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
out_channels = 128  
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='npu')]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]