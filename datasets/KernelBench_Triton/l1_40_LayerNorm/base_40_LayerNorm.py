import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _layernorm_partial_sums_kernel(
    x_ptr,
    partial_sums_ptr,
    partial_sumsq_ptr,
    M,
    PARTS_PER_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row = pid // PARTS_PER_ROW
    part = pid % PARTS_PER_ROW

    block_idx = part
    acc = 0.0
    acc2 = 0.0
    while block_idx * BLOCK_SIZE < M:
        col_start = block_idx * BLOCK_SIZE
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(x_ptr + row * M + offsets, mask=mask,
                    other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
        acc2 += tl.sum(x * x, axis=0)
        block_idx += PARTS_PER_ROW

    out_idx = row * PARTS_PER_ROW + part
    tl.store(partial_sums_ptr + out_idx, acc)
    tl.store(partial_sumsq_ptr + out_idx, acc2)


@triton.jit
def _layernorm_stats_from_partials_kernel(
    partial_sums_ptr,
    partial_sumsq_ptr,
    mean_ptr,
    rstd_ptr,
    INV_M,
    EPSILON,
    PARTS_PER_ROW: tl.constexpr,
):
    row = tl.program_id(axis=0)
    part_offsets = tl.arange(0, PARTS_PER_ROW)
    base = row * PARTS_PER_ROW + part_offsets
    sums = tl.load(partial_sums_ptr + base)
    sums2 = tl.load(partial_sumsq_ptr + base)
    total = tl.sum(sums, axis=0)
    total2 = tl.sum(sums2, axis=0)
    mean = total * INV_M
    var = total2 * INV_M - mean * mean
    rstd = tl.rsqrt(var + EPSILON)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layernorm_apply_partitioned_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    y_ptr,
    M,
    PARTS_PER_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row = pid // PARTS_PER_ROW
    part = pid % PARTS_PER_ROW

    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    block_idx = part
    while block_idx * BLOCK_SIZE < M:
        col_start = block_idx * BLOCK_SIZE
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
        mask = offsets < M

        x = tl.load(x_ptr + row * M + offsets, mask=mask,
                    other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd
        y = y * w + b
        tl.store(y_ptr + row * M + offsets, y, mask=mask)
        block_idx += PARTS_PER_ROW


def _pick_configs(M: int, rows: int) -> tuple[int, int, int]:
    if M >= 1 << 22:
        return 12, 8192, 8192
    if M >= 1 << 20:
        return 8, 4096, 2048
    if M >= 1 << 18:
        return 8, 2048, 1024
    if M >= 1 << 16:
        return 4, 2048, 1024
    parts = 1 if rows >= 64 else 2
    block = min(1024, triton.next_power_of_2(M))
    return parts, block, block


def _layer_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    normalized_shape: tuple,
    eps: float,
):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape, )
    M = math.prod(normalized_shape)
    rows = x.numel() // M

    x_in = x.contiguous().view(rows, M)
    y_out = torch.empty_like(x_in)
    w = weight.contiguous().view(M)
    b = bias.contiguous().view(M)

    parts_per_row, reduce_block, apply_block = _pick_configs(M, rows)
    partial_count = rows * parts_per_row
    partial_sums = torch.empty(partial_count,
                               device=x.device,
                               dtype=torch.float32)
    partial_sumsq = torch.empty_like(partial_sums)
    mean = torch.empty(rows, device=x.device, dtype=torch.float32)
    rstd = torch.empty(rows, device=x.device, dtype=torch.float32)

    reduction_grid = (partial_count, )
    _layernorm_partial_sums_kernel[reduction_grid](
        x_in,
        partial_sums,
        partial_sumsq,
        M,
        PARTS_PER_ROW=parts_per_row,
        BLOCK_SIZE=reduce_block,
        num_warps=8,
        num_stages=4,
    )

    _layernorm_stats_from_partials_kernel[(rows, )](
        partial_sums,
        partial_sumsq,
        mean,
        rstd,
        float(1.0 / M),
        float(eps),
        PARTS_PER_ROW=parts_per_row,
    )

    _layernorm_apply_partitioned_kernel[reduction_grid](
        x_in,
        w,
        b,
        mean,
        rstd,
        y_out,
        M,
        PARTS_PER_ROW=parts_per_row,
        BLOCK_SIZE=apply_block,
        num_warps=8,
        num_stages=4,
    )

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
