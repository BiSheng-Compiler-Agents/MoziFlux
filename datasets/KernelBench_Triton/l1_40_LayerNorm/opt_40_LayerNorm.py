import math
import torch
import torch.nn as nn

import triton
import triton.language as tl
import triton.runtime.driver as driver

_MAX_PROGRAMS = 65535
_REDUCE_BLOCK = 16384
_APPLY_BLOCK = 8192
_MAX_REDUCE_TILES = 8192


def _ceil_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _get_vector_cores(x: torch.Tensor) -> int:
    try:
        return int(
            driver.active.utils.get_device_properties(
                x.device)["num_vectorcore"])
    except Exception:
        try:
            return int(
                driver.active.utils.get_device_properties("npu")
                ["num_vectorcore"])
        except Exception:
            return 24


@triton.jit
def _layernorm_partial_kernel(
    x_ptr,
    partial_sums_ptr,
    partial_sumsq_ptr,
    M,
    num_tiles: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // num_tiles
    tile = pid - row * num_tiles
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)

    x = tl.load(x_ptr + row * M + offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    s2 = tl.sum(x * x, axis=0)
    out_idx = row * num_tiles + tile
    tl.store(partial_sums_ptr + out_idx, s)
    tl.store(partial_sumsq_ptr + out_idx, s2)


@triton.jit
def _layernorm_partial_persistent_kernel(
    x_ptr,
    partial_sums_ptr,
    partial_sumsq_ptr,
    M,
    total_tasks,
    n_programs,
    num_tiles: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for task in range(pid, total_tasks, n_programs):
        row = task // num_tiles
        tile = task - row * num_tiles
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        tl.multiple_of(offsets, 16)
        tl.max_contiguous(offsets, BLOCK_SIZE)
        x = tl.load(x_ptr + row * M + offsets, mask=mask,
                    other=0.0).to(tl.float32)
        s = tl.sum(x, axis=0)
        s2 = tl.sum(x * x, axis=0)
        out_idx = row * num_tiles + tile
        tl.store(partial_sums_ptr + out_idx, s)
        tl.store(partial_sumsq_ptr + out_idx, s2)


@triton.jit
def _layernorm_finalize_kernel(
    partial_sums_ptr,
    partial_sumsq_ptr,
    mean_ptr,
    rstd_ptr,
    rows,
    INV_M,
    EPSILON,
    n_programs,
    num_tiles: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    tile_mask = tile_offsets < num_tiles
    for row in range(pid, rows, n_programs):
        base = row * num_tiles + tile_offsets
        ps = tl.load(partial_sums_ptr + base, mask=tile_mask,
                     other=0.0).to(tl.float32)
        ps2 = tl.load(partial_sumsq_ptr + base, mask=tile_mask,
                      other=0.0).to(tl.float32)
        s = tl.sum(ps, axis=0)
        s2 = tl.sum(ps2, axis=0)
        mean = s * INV_M
        var = s2 * INV_M - mean * mean
        var = tl.maximum(var, 0.0)
        rstd = tl.rsqrt(var + EPSILON)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layernorm_apply_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    y_ptr,
    M,
    num_tiles: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // num_tiles
    tile = pid - row * num_tiles
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)

    idx = row * M + offsets
    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)
    y = (x - mean) * rstd
    y = y * w + b
    tl.store(y_ptr + idx, y, mask=mask)


@triton.jit
def _layernorm_apply_persistent_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    y_ptr,
    M,
    total_tasks,
    n_programs,
    num_tiles: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for task in range(pid, total_tasks, n_programs):
        row = task // num_tiles
        tile = task - row * num_tiles
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        tl.multiple_of(offsets, 16)
        tl.max_contiguous(offsets, BLOCK_SIZE)
        idx = row * M + offsets
        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)
        y = (x - mean) * rstd
        y = y * w + b
        tl.store(y_ptr + idx, y, mask=mask)


def _layer_norm_triton(x: torch.Tensor, weight: torch.Tensor,
                       bias: torch.Tensor, normalized_shape: tuple,
                       eps: float):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape, )
    M = math.prod(normalized_shape)
    rows = x.numel() // M
    if rows == 0 or M == 0:
        return torch.empty_like(x)

    x_in = x.contiguous().view(rows, M)
    y_out = torch.empty_like(x_in)
    w = weight.contiguous().view(M)
    b = bias.contiguous().view(M)

    reduce_block_size = _REDUCE_BLOCK
    apply_block_size = _APPLY_BLOCK
    reduce_tiles = triton.cdiv(M, reduce_block_size)
    apply_tiles = triton.cdiv(M, apply_block_size)
    if reduce_tiles > _MAX_REDUCE_TILES:
        return torch.nn.functional.layer_norm(x, normalized_shape, weight,
                                              bias, eps)

    reduce_tasks = rows * reduce_tiles
    apply_tasks = rows * apply_tiles
    partial_sums = torch.empty((rows, reduce_tiles),
                               device=x.device,
                               dtype=torch.float32)
    partial_sumsq = torch.empty((rows, reduce_tiles),
                                device=x.device,
                                dtype=torch.float32)
    mean = torch.empty((rows, ), device=x.device, dtype=torch.float32)
    rstd = torch.empty((rows, ), device=x.device, dtype=torch.float32)

    if reduce_tasks <= _MAX_PROGRAMS:
        _layernorm_partial_kernel[(reduce_tasks, )](
            x_in,
            partial_sums,
            partial_sumsq,
            M,
            reduce_tiles,
            BLOCK_SIZE=reduce_block_size)
    else:
        n_programs = _MAX_PROGRAMS
        _layernorm_partial_persistent_kernel[(n_programs, )](
            x_in,
            partial_sums,
            partial_sumsq,
            M,
            reduce_tasks,
            n_programs,
            reduce_tiles,
            BLOCK_SIZE=reduce_block_size)

    reduce_block = _ceil_pow2(reduce_tiles)
    n_finalize = max(1, min(rows, _MAX_PROGRAMS, _get_vector_cores(x)))
    inv_m = float(1.0 / M)
    _layernorm_finalize_kernel[(n_finalize, )](partial_sums,
                                               partial_sumsq,
                                               mean,
                                               rstd,
                                               rows,
                                               inv_m,
                                               float(eps),
                                               n_finalize,
                                               reduce_tiles,
                                               BLOCK_TILES=reduce_block)

    if apply_tasks <= _MAX_PROGRAMS:
        _layernorm_apply_kernel[(apply_tasks, )](x_in,
                                                 w,
                                                 b,
                                                 mean,
                                                 rstd,
                                                 y_out,
                                                 M,
                                                 apply_tiles,
                                                 BLOCK_SIZE=apply_block_size)
    else:
        n_programs = _MAX_PROGRAMS
        _layernorm_apply_persistent_kernel[(n_programs, )](
            x_in,
            w,
            b,
            mean,
            rstd,
            y_out,
            M,
            apply_tasks,
            n_programs,
            apply_tiles,
            BLOCK_SIZE=apply_block_size)
    return y_out.view_as(x)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    normalized_shape,
    eps: float = 1e-5,
) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("layer_norm requires an Ascend NPU tensor input")
    return _layer_norm_triton(x, weight, bias, normalized_shape, eps)


class ModelNew(nn.Module):
    """LayerNorm optimized for Ascend NPU with non-atomic partial reductions."""

    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(
            torch.ones(self.normalized_shape, dtype=torch.float32))
        self.bias = nn.Parameter(
            torch.zeros(self.normalized_shape, dtype=torch.float32))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(x, self.weight, self.bias, self.normalized_shape,
                          self.eps)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [(features, dim1, dim2)]
