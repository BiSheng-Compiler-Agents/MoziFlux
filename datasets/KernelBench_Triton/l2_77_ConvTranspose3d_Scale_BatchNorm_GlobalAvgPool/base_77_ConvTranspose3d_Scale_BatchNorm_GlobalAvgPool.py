import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

_TARGET_BATCH = 16
_TARGET_OUT_CHANNELS = 128
_TARGET_POOL_DHW = (20, 36, 36)
_TARGET_POOL_L = _TARGET_POOL_DHW[0] * _TARGET_POOL_DHW[1] * _TARGET_POOL_DHW[2]
_TARGET_POOL_M = _TARGET_BATCH * _TARGET_OUT_CHANNELS
_TARGET_BLOCK_M = 8
_TARGET_BLOCK_N = 1536
_USE_TARGET_PREFETCH = True


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 2,
            "BLOCK_N": 512
        },
                      num_warps=4,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 4,
            "BLOCK_N": 512
        },
                      num_warps=4,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 4,
            "BLOCK_N": 1024
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 8,
            "BLOCK_N": 1024
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 4,
            "BLOCK_N": 2048
        },
                      num_warps=8,
                      num_stages=5),
    ],
    key=["L"],
)
@triton.jit
def _global_avg_pool3d_ncdhw_kernel(
    x_ptr,
    y_ptr,
    M,
    L: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M
    acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    for start in range(0, L, BLOCK_N):
        idx = start + cols
        mask = row_mask[:, None] & (idx[None, :] < L)
        ptrs = x_ptr + rows[:, None] * L + idx[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float32), axis=1)

    tl.store(y_ptr + rows, acc * (1.0 / L), mask=row_mask)


@triton.jit
def _global_avg_pool3d_target_kernel(
    x_ptr,
    y_ptr,
    TARGET_L: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    for start in range(0, TARGET_L, BLOCK_N):
        idx = start + cols
        mask = idx[None, :] < TARGET_L
        ptrs = x_ptr + rows[:, None] * TARGET_L + idx[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float32), axis=1)

    tl.store(y_ptr + rows, acc * (1.0 / TARGET_L))


@triton.jit
def _global_avg_pool3d_target_prefetch_kernel(
    x_ptr,
    y_ptr,
    TARGET_L: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    idx = cols
    mask = idx[None, :] < TARGET_L
    ptrs = x_ptr + rows[:, None] * TARGET_L + idx[None, :]
    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    for start in range(BLOCK_N, TARGET_L, BLOCK_N):
        idx_next = start + cols
        mask_next = idx_next[None, :] < TARGET_L
        ptrs_next = x_ptr + rows[:, None] * TARGET_L + idx_next[None, :]
        next_vals = tl.load(ptrs_next, mask=mask_next,
                            other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=1)
        vals = next_vals

    acc += tl.sum(vals, axis=1)
    tl.store(y_ptr + rows, acc * (1.0 / TARGET_L))


def global_avg_pool3d_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 5, "Input must be NCDHW"
    N, C, D, H, W = x.shape
    M = N * C
    L = D * H * W

    x_contig = x.contiguous()
    y = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    x_flat = x_contig.view(-1)
    y_flat = y.view(-1)

    if M == _TARGET_POOL_M and L == _TARGET_POOL_L:
        grid = (triton.cdiv(M, _TARGET_BLOCK_M), )
        if _USE_TARGET_PREFETCH:
            _global_avg_pool3d_target_prefetch_kernel[grid](
                x_flat,
                y_flat,
                TARGET_L=_TARGET_POOL_L,
                BLOCK_M=_TARGET_BLOCK_M,
                BLOCK_N=_TARGET_BLOCK_N,
            )
        else:
            _global_avg_pool3d_target_kernel[grid](
                x_flat,
                y_flat,
                TARGET_L=_TARGET_POOL_L,
                BLOCK_M=_TARGET_BLOCK_M,
                BLOCK_N=_TARGET_BLOCK_N,
            )
    else:

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), )

        _global_avg_pool3d_ncdhw_kernel[grid](
            x_flat,
            y_flat,
            M,
            L=L,
        )
    return y


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


class ModelNew(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 scale_factor,
                 eps=1e-5,
                 momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                 kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels,
                                         eps=eps,
                                         momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        w = self.conv_transpose.weight * self.scale_factor
        b = None if self.conv_transpose.bias is None else (
            self.conv_transpose.bias * self.scale_factor)
        x = F.conv_transpose3d(
            x,
            w,
            bias=b,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            groups=self.conv_transpose.groups,
            dilation=self.conv_transpose.dilation,
        )

        use_commute = (not self.batch_norm.training) and getattr(
            self.batch_norm, "track_running_stats", True)
        if use_commute:
            if _is_npu_tensor(x):
                x = global_avg_pool3d_triton(x)
            else:
                raise RuntimeError(
                    "ModelNew expects Ascend NPU tensors for the Triton pooling path."
                )
            dtype = x.dtype
            rm = self.batch_norm.running_mean.to(dtype).view(1, -1, 1, 1, 1)
            rv = self.batch_norm.running_var.to(dtype).view(1, -1, 1, 1, 1)
            inv_std = torch.rsqrt(rv + self.batch_norm.eps)

            if self.batch_norm.affine:
                weight = self.batch_norm.weight.to(dtype).view(1, -1, 1, 1, 1)
                bias = self.batch_norm.bias.to(dtype).view(1, -1, 1, 1, 1)
            else:
                weight = torch.ones(1,
                                    x.size(1),
                                    1,
                                    1,
                                    1,
                                    device=x.device,
                                    dtype=dtype)
                bias = torch.zeros(1,
                                   x.size(1),
                                   1,
                                   1,
                                   1,
                                   device=x.device,
                                   dtype=dtype)

            x = (x - rm) * (weight * inv_std) + bias
            return x
        else:
            x = self.batch_norm(x)
            if _is_npu_tensor(x):
                x = global_avg_pool3d_triton(x)
            else:
                raise RuntimeError(
                    "ModelNew expects Ascend NPU tensors for the Triton pooling path."
                )
            return x


batch_size = 16
in_channels = 64
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
scale_factor = 2.0
_MODEL_CACHE = {}


def _get_default_model(device: torch.device, dtype: torch.dtype) -> ModelNew:
    cache_key = (device.type, getattr(device, "index", None), dtype)
    if cache_key not in _MODEL_CACHE:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = ModelNew(*get_init_inputs())
        model = model.to(device=device, dtype=dtype)
        model.eval()
        _MODEL_CACHE[cache_key] = model
    return _MODEL_CACHE[cache_key]


def conv_transpose3d_scale_batch_norm_global_avg_pool(
        x: torch.Tensor) -> torch.Tensor:
    model = _get_default_model(x.device, x.dtype)
    with torch.no_grad():
        return model(x)


batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
