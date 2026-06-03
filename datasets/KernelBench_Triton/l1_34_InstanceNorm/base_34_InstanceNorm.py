import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _instance_norm2d_kernel(
    x_ptr,
    y_ptr,
    NC,
    eps,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row = pid

    x_base = x_ptr + row * HW
    y_base = y_ptr + row * HW

    idx_vec = tl.arange(0, BLOCK_SIZE)

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    idx0 = idx_vec
    mask0 = idx0 < HW
    x0 = tl.load(x_base + idx0, mask=mask0, other=0.0)
    xf0 = x0.to(tl.float32)

    for off in tl.static_range(BLOCK_SIZE, HW, BLOCK_SIZE):
        idx1 = off + idx_vec
        mask1 = idx1 < HW
        x1 = tl.load(x_base + idx1, mask=mask1, other=0.0)

        sum_val += tl.sum(xf0, axis=0)
        sumsq_val += tl.sum(xf0 * xf0, axis=0)

        idx0, mask0, x0 = idx1, mask1, x1
        xf0 = x0.to(tl.float32)

    sum_val += tl.sum(xf0, axis=0)
    sumsq_val += tl.sum(xf0 * xf0, axis=0)

    inv_hw = 1.0 / tl.full((), HW, dtype=tl.float32)
    mean = sum_val * inv_hw
    var = sumsq_val * inv_hw - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)

    idx0 = idx_vec
    mask0 = idx0 < HW
    x0 = tl.load(x_base + idx0, mask=mask0, other=0.0)
    xf0 = x0.to(tl.float32)

    for off in tl.static_range(BLOCK_SIZE, HW, BLOCK_SIZE):
        idx1 = off + idx_vec
        mask1 = idx1 < HW
        x1 = tl.load(x_base + idx1, mask=mask1, other=0.0)

        y0 = (xf0 - mean) * rstd
        tl.store(y_base + idx0, y0.to(x0.dtype), mask=mask0)

        idx0, mask0, x0 = idx1, mask1, x1
        xf0 = x0.to(tl.float32)

    y_last = (xf0 - mean) * rstd
    tl.store(y_base + idx0, y_last.to(x0.dtype), mask=mask0)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "Expected input of shape (N, C, H, W)"
        _, C, _, _ = x.shape
        assert C == self.num_features, f"Expected C == num_features ({self.num_features}), got {C}"
        return instance_norm_2d(x, eps=self.eps)


def instance_norm_2d(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.dim() == 4, "Expected input of shape (N, C, H, W)"
    assert x.device.type in {"cuda", "npu"}, "InstanceNorm Triton kernel requires an accelerator tensor"

    x = x.contiguous()
    y = torch.empty_like(x)

    N, C, H, W = x.shape
    HW = H * W
    NC = N * C
    grid = (NC,)

    _instance_norm2d_kernel[grid](
        x,
        y,
        NC,
        eps,
        HW=HW,
        BLOCK_SIZE=8192,
        num_warps=8,
        num_stages=2,
    )
    return y


batch_size = 112
features = 64
dim1 = 512
dim2 = 512


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features]
