import torch
import torch.nn as nn
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK_SIZE = 2048


@triton.jit
def _instancenorm_single_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    eps,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid - n * C
    x_base = x_ptr + n * stride_n + c * stride_c
    y_base = y_ptr + n * stride_n + c * stride_c
    idx_vec = tl.arange(0, BLOCK_SIZE)

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)
    off0 = 0
    idx0 = off0 + idx_vec
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

    off0 = 0
    idx0 = off0 + idx_vec
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


@triton.jit
def _instancenorm_partial_direct(
    x_ptr,
    partial_sum_ptr,
    partial_sumsq_ptr,
    HW: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    plane = pid // NUM_BLOCKS
    block = pid - plane * NUM_BLOCKS
    offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
    mask = offs < HW
    vals = tl.load(x_ptr + plane * HW + offs, mask=mask,
                   other=0.0).to(tl.float32)
    ss = tl.sum(vals, axis=0)
    sq = tl.sum(vals * vals, axis=0)
    tl.store(partial_sum_ptr + pid, ss, mask=True)
    tl.store(partial_sumsq_ptr + pid, sq, mask=True)


@triton.jit
def _instancenorm_partial_persistent(
    x_ptr,
    partial_sum_ptr,
    partial_sumsq_ptr,
    n_programs,
    TOTAL_TILES: tl.constexpr,
    HW: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for tile in range(pid, TOTAL_TILES, n_programs):
        plane = tile // NUM_BLOCKS
        block = tile - plane * NUM_BLOCKS
        offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
        mask = offs < HW
        vals = tl.load(x_ptr + plane * HW + offs, mask=mask,
                       other=0.0).to(tl.float32)
        ss = tl.sum(vals, axis=0)
        sq = tl.sum(vals * vals, axis=0)
        tl.store(partial_sum_ptr + tile, ss, mask=True)
        tl.store(partial_sumsq_ptr + tile, sq, mask=True)


@triton.jit
def _instancenorm_finalize_direct(
    partial_sum_ptr,
    partial_sumsq_ptr,
    scale_ptr,
    shift_ptr,
    eps,
    HW: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_PARTS: tl.constexpr,
):
    plane = tl.program_id(0)
    offs = tl.arange(0, BLOCK_PARTS)
    mask = offs < NUM_BLOCKS
    base = plane * NUM_BLOCKS + offs
    sums = tl.load(partial_sum_ptr + base, mask=mask, other=0.0)
    sumsqs = tl.load(partial_sumsq_ptr + base, mask=mask, other=0.0)
    sum_val = tl.sum(sums, axis=0)
    sumsq_val = tl.sum(sumsqs, axis=0)
    inv_hw = 1.0 / tl.full((), HW, dtype=tl.float32)
    mean = sum_val * inv_hw
    var = sumsq_val * inv_hw - mean * mean
    var = tl.maximum(var, 0.0)
    scale = tl.rsqrt(var + eps)
    shift = -mean * scale
    tl.store(scale_ptr + plane, scale, mask=True)
    tl.store(shift_ptr + plane, shift, mask=True)


@triton.jit
def _instancenorm_finalize_persistent(
    partial_sum_ptr,
    partial_sumsq_ptr,
    scale_ptr,
    shift_ptr,
    n_programs,
    eps,
    TOTAL_PLANES: tl.constexpr,
    HW: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_PARTS: tl.constexpr,
):
    pid = tl.program_id(0)
    for plane in range(pid, TOTAL_PLANES, n_programs):
        offs = tl.arange(0, BLOCK_PARTS)
        mask = offs < NUM_BLOCKS
        base = plane * NUM_BLOCKS + offs
        sums = tl.load(partial_sum_ptr + base, mask=mask, other=0.0)
        sumsqs = tl.load(partial_sumsq_ptr + base, mask=mask, other=0.0)
        sum_val = tl.sum(sums, axis=0)
        sumsq_val = tl.sum(sumsqs, axis=0)
        inv_hw = 1.0 / tl.full((), HW, dtype=tl.float32)
        mean = sum_val * inv_hw
        var = sumsq_val * inv_hw - mean * mean
        var = tl.maximum(var, 0.0)
        scale = tl.rsqrt(var + eps)
        shift = -mean * scale
        tl.store(scale_ptr + plane, scale, mask=True)
        tl.store(shift_ptr + plane, shift, mask=True)


@triton.jit
def _instancenorm_apply_direct(
    x_ptr,
    y_ptr,
    scale_ptr,
    shift_ptr,
    HW: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    plane = pid // NUM_BLOCKS
    block = pid - plane * NUM_BLOCKS
    offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
    mask = offs < HW
    scale = tl.load(scale_ptr + plane, mask=True, other=1.0)
    shift = tl.load(shift_ptr + plane, mask=True, other=0.0)
    x = tl.load(x_ptr + plane * HW + offs, mask=mask, other=0.0)
    y = x.to(tl.float32) * scale + shift
    tl.store(y_ptr + plane * HW + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _instancenorm_apply_persistent(
    x_ptr,
    y_ptr,
    scale_ptr,
    shift_ptr,
    n_programs,
    TOTAL_TILES: tl.constexpr,
    HW: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for tile in range(pid, TOTAL_TILES, n_programs):
        plane = tile // NUM_BLOCKS
        block = tile - plane * NUM_BLOCKS
        offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
        mask = offs < HW
        scale = tl.load(scale_ptr + plane, mask=True, other=1.0)
        shift = tl.load(shift_ptr + plane, mask=True, other=0.0)
        x = tl.load(x_ptr + plane * HW + offs, mask=mask, other=0.0)
        y = x.to(tl.float32) * scale + shift
        tl.store(y_ptr + plane * HW + offs, y.to(x.dtype), mask=mask)


class ModelNew(nn.Module):
    """InstanceNorm2d (affine=False, track_running_stats=False) optimized for large HW planes."""

    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "Expected input of shape (N, C, H, W)"
        _, C, _, _ = x.shape
        assert C == self.num_features, f"Expected C == num_features ({self.num_features}), got {C}"
        return instance_norm_2d(x, eps=self.eps)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def instance_norm_2d(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.dim() == 4, "Expected input of shape (N, C, H, W)"
    assert x.device.type in {
        "cuda", "npu"
    }, "InstanceNorm Triton kernel requires an accelerator tensor"

    x = x.contiguous()
    y = torch.empty_like(x)
    N, C, H, W = x.shape
    HW = H * W
    total_planes = N * C
    num_blocks = triton.cdiv(HW, _BLOCK_SIZE)
    total_tiles = total_planes * num_blocks
    block_parts = _next_power_of_2(num_blocks)

    if total_tiles <= _MAX_PROGRAMS:
        stride_n, stride_c, stride_h, stride_w = x.stride()
        _instancenorm_single_kernel[(total_planes, )](
            x,
            y,
            N,
            C,
            H,
            W,
            stride_n,
            stride_c,
            stride_h,
            stride_w,
            eps,
            HW=HW,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
        return y

    partial_sum = torch.empty((total_planes, num_blocks),
                              device=x.device,
                              dtype=torch.float32)
    partial_sumsq = torch.empty((total_planes, num_blocks),
                                device=x.device,
                                dtype=torch.float32)
    scale = torch.empty((total_planes, ), device=x.device, dtype=torch.float32)
    shift = torch.empty((total_planes, ), device=x.device, dtype=torch.float32)

    if total_tiles > _MAX_PROGRAMS:
        n_programs = _MAX_PROGRAMS
        _instancenorm_partial_persistent[(n_programs, )](
            x,
            partial_sum,
            partial_sumsq,
            n_programs,
            total_tiles,
            HW,
            num_blocks,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
    else:
        _instancenorm_partial_direct[(total_tiles, )](
            x,
            partial_sum,
            partial_sumsq,
            HW,
            num_blocks,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )

    if total_planes > _MAX_PROGRAMS:
        n_programs = _MAX_PROGRAMS
        _instancenorm_finalize_persistent[(n_programs, )](
            partial_sum,
            partial_sumsq,
            scale,
            shift,
            n_programs,
            eps,
            total_planes,
            HW,
            num_blocks,
            block_parts,
            num_warps=4,
            num_stages=2,
        )
    else:
        _instancenorm_finalize_direct[(total_planes, )](
            partial_sum,
            partial_sumsq,
            scale,
            shift,
            eps,
            HW,
            num_blocks,
            block_parts,
            num_warps=4,
            num_stages=2,
        )

    if total_tiles > _MAX_PROGRAMS:
        n_programs = _MAX_PROGRAMS
        _instancenorm_apply_persistent[(n_programs, )](
            x,
            y,
            scale,
            shift,
            n_programs,
            total_tiles,
            HW,
            num_blocks,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
    else:
        _instancenorm_apply_direct[(total_tiles, )](
            x,
            y,
            scale,
            shift,
            HW,
            num_blocks,
            BLOCK_SIZE=_BLOCK_SIZE,
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
