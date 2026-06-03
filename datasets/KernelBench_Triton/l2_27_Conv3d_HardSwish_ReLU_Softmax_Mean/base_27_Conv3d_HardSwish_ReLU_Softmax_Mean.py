import torch
import torch.nn as nn
import triton
import triton.language as tl

_MODEL_CACHE = {}
TARGET_CHANNELS = 16
TUNE_BLOCK_S = 320
TUNE_NUM_WARPS = 8
TUNE_NUM_STAGES = 2


@triton.jit
def _fused_hswish_relu_softmax_mean_kernel(
    x_ptr,             # *float or *half, shape [N, C, S] (S = D*H*W), contiguous N,C,S layout
    out_ptr,           # *float or *half, shape [N, C]
    N, C, S,           # int32
    stride_n,          # int32, elements between successive n
    stride_c,          # int32, elements between successive c
    stride_out_n,      # int32, elements between successive n in out
    inv_S,             # float32, 1.0 / S
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    base_n = pid_n * stride_n
    c_idx = tl.arange(0, BLOCK_C)
    valid_c = c_idx < C
    c_base = base_n + c_idx[:, None] * stride_c

    # Accumulator over spatial positions for each channel
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    inv6 = 1.0 / 6.0
    s_start = 0
    while s_start < S:
        s_idx = s_start + tl.arange(0, BLOCK_S)
        valid_s = s_idx < S
        mask = valid_c[:, None] & valid_s[None, :]

        # Offsets for 2D tile [C, S_tile]
        offs = c_base + s_idx[None, :]

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # Fused HardSwish + ReLU: y = max(x, 0) * clamp(x + 3, 0, 6) / 6
        t = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        y = tl.maximum(x, 0.0) * (t * inv6)

        # Mask invalid spatial lanes with -inf for softmax max-reduction
        y_masked = tl.where(mask, y, -float("inf"))

        # Softmax across channels (axis=0) per spatial column
        m = tl.max(y_masked, axis=0)                           # [BLOCK_S]
        expv = tl.exp(y_masked - m[None, :])                   # [BLOCK_C, BLOCK_S]
        sumexp = tl.sum(expv, axis=0)                          # [BLOCK_S]
        p = expv / sumexp[None, :]                             # [BLOCK_C, BLOCK_S]
        p = tl.where(mask, p, 0.0)

        # Accumulate probabilities across spatial positions for each channel
        acc += tl.sum(p, axis=1)

        s_start += BLOCK_S

    # Write normalized mean over spatial dims
    out_offs = pid_n * stride_out_n + c_idx
    tl.store(out_ptr + out_offs, acc * inv_S, mask=valid_c)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def fused_hswish_relu_softmax_mean(x: torch.Tensor) -> torch.Tensor:
    # x: [N, C, D, H, W]
    if x.device.type != "npu":
        raise RuntimeError("fused_hswish_relu_softmax_mean expects an Ascend NPU tensor")
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()

    # Strides in elements for a contiguous [N, C, S] view
    stride_c = S
    stride_n = C * S

    # Output [N, C], same dtype as input
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)
    stride_out_n = C

    if C != TARGET_CHANNELS:
        raise RuntimeError(
            f"fused_hswish_relu_softmax_mean expects C == {TARGET_CHANNELS}, got {C}"
        )

    BLOCK_S = TUNE_BLOCK_S

    grid = (N,)
    inv_S = float(1.0 / S)

    _fused_hswish_relu_softmax_mean_kernel[grid](
        x, out,
        N, C, S,
        stride_n, stride_c, stride_out_n,
        inv_S,
        BLOCK_C=TARGET_CHANNELS,
        BLOCK_S=BLOCK_S,
        num_warps=TUNE_NUM_WARPS,
        num_stages=TUNE_NUM_STAGES,
    )
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies HardSwish, ReLU, Softmax, and then calculates the mean.
    Fused with a Triton kernel for post-convolution operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        # Conv stays in PyTorch (highly optimized), post-ops fused in Triton
        x = self.conv(x)
        return fused_hswish_relu_softmax_mean(x)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3


def _set_deterministic_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def conv3d_hardswish_relu_softmax_mean(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("conv3d_hardswish_relu_softmax_mean expects an Ascend NPU tensor")

    key = (str(x.device), x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        _set_deterministic_seed(0)
        model = ModelNew(*get_init_inputs()).eval().to(device=x.device, dtype=x.dtype)
        _MODEL_CACHE[key] = model

    with torch.no_grad():
        return model(x)


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
