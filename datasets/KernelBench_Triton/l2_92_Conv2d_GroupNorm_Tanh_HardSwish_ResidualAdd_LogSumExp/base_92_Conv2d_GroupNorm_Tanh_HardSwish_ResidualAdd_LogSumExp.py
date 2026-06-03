import math
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 8
DEFAULT_OUT_CHANNELS = 64
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_KERNEL_SIZE = 3
DEFAULT_GROUPS = 16


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _fused_tanh_hswish_residual_lse(
    x_conv_ptr,
    x_norm_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    HW = H * W
    total = N * HW

    mask_pid = pid < total

    n = pid // HW
    q = pid % HW
    base = n * C * HW + q

    arange_c = tl.arange(0, BLOCK_C)

    m = -1.0e30
    sum_exp = 0.0

    c0 = 0
    while c0 < C:
        idx = c0 + arange_c
        ch_mask = (idx < C) & mask_pid
        offs = base + idx * HW

        xc = tl.load(x_conv_ptr + offs, mask=ch_mask, other=-1.0e30).to(tl.float32)
        xn = tl.load(x_norm_ptr + offs, mask=ch_mask, other=0.0).to(tl.float32)

        t2 = tl.exp(-2.0 * xn)
        t = (1.0 - t2) / (1.0 + t2)
        hsw = t * (t + 3.0) * (1.0 / 6.0)

        y = xc + hsw

        tile_max = tl.max(y, axis=0)
        m_new = tl.where(tile_max > m, tile_max, m)
        sum_exp = sum_exp * tl.exp(m - m_new)
        e = tl.exp(y - m_new)
        e = tl.where(ch_mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)
        m = m_new

        c0 += BLOCK_C

    lse = m + tl.log(sum_exp)
    tl.store(out_ptr + pid, lse, mask=mask_pid)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        groups=DEFAULT_GROUPS,
        eps=1e-5,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")

        x_conv = self.conv(x)
        x_norm = self.group_norm(x_conv)

        N, C, H, W = x_conv.shape
        x_conv_c = x_conv.contiguous()
        x_norm_c = x_norm.contiguous()
        out = torch.empty((N, 1, H, W), device=x_conv.device, dtype=x_conv.dtype)

        if C <= 1:
            block_c = 1
        else:
            block_c = 1 << int(math.ceil(math.log2(C)))
            block_c = min(128, max(1, block_c))

        HW = H * W
        grid = (HW,)
        for n in range(N):
            xc_slice = x_conv_c[n:n+1].contiguous()
            xn_slice = x_norm_c[n:n+1].contiguous()
            out_slice = out[n:n+1].contiguous()
            _fused_tanh_hswish_residual_lse[grid](
                xc_slice, xn_slice, out_slice,
                1, C, H, W,
                BLOCK_C=block_c,
                num_warps=2,
                num_stages=2,
            )
        return out


_MODEL_CACHE: dict[tuple[torch.device, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().eval().to(device=x.device, dtype=x.dtype)
        _MODEL_CACHE[key] = model
    return model(x)
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='npu')]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]
