import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _argmin_row_kernel(
    x_ptr,
    out_ptr,
    rows,
    cols,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < rows
    k_offsets = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K),
        BLOCK_K,
    )[None, :]
    row_bases = row_offsets[:, None] * cols

    best_val = tl.full((BLOCK_M, ), float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_M, ), dtype=tl.int32)

    k0 = 0
    while k0 < cols:
        current_k = k0 + k_offsets
        mask = row_mask[:, None] & (current_k < cols)
        values = tl.load(
            x_ptr + row_bases + current_k,
            mask=mask,
            other=float("inf"),
            cache_modifier=".cg",
        ).to(tl.float32)

        tile_min = tl.min(values, axis=1)
        equal_mask = (values == tile_min[:, None]) & mask
        invalid_index = tl.full((BLOCK_M, BLOCK_K), cols, dtype=tl.int32)
        tile_indices = tl.where(equal_mask, current_k.to(tl.int32),
                                invalid_index)
        tile_first_idx = tl.min(tile_indices, axis=1)

        should_update = (tile_min < best_val) | ((tile_min == best_val) &
                                                 (tile_first_idx < best_idx))
        best_val = tl.where(should_update, tile_min, best_val)
        best_idx = tl.where(should_update, tile_first_idx, best_idx)
        k0 += BLOCK_K

    tl.store(out_ptr + row_offsets, best_idx, mask=row_mask)


@triton.jit
def _argmin_row_kernel_full_tiles(
    x_ptr,
    out_ptr,
    rows,
    cols,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < rows
    k_offsets = tl.arange(0, BLOCK_K)[None, :]
    row_bases = row_offsets[:, None] * cols

    best_val = tl.full((BLOCK_M, ), float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_M, ), dtype=tl.int32)
    invalid_index = tl.full((BLOCK_M, BLOCK_K), cols, dtype=tl.int32)
    x_ptrs = x_ptr + row_bases + k_offsets
    tile_offsets = k_offsets.to(tl.int32)

    k0 = 0
    while k0 < cols:
        values = tl.load(
            x_ptrs,
            mask=row_mask[:, None],
            other=float("inf"),
        ).to(tl.float32)

        tile_min = tl.min(values, axis=1)
        tile_indices = tl.where(values == tile_min[:, None], tile_offsets,
                                invalid_index)
        tile_first_idx = tl.min(tile_indices, axis=1)

        should_update = (tile_min < best_val) | ((tile_min == best_val) &
                                                 (tile_first_idx < best_idx))
        best_val = tl.where(should_update, tile_min, best_val)
        best_idx = tl.where(should_update, tile_first_idx, best_idx)
        x_ptrs += BLOCK_K
        tile_offsets += BLOCK_K
        k0 += BLOCK_K

    tl.store(out_ptr + row_offsets, best_idx, mask=row_mask)


@triton.jit
def _argmin_row_kernel_4096(
    x_ptr,
    out_ptr,
    rows,
    cols,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < rows
    k_offsets = tl.arange(0, BLOCK_K)[None, :]
    row_bases = row_offsets[:, None] * cols

    best_val = tl.full((BLOCK_M, ), float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_M, ), dtype=tl.int32)
    invalid_index = tl.full((BLOCK_M, BLOCK_K), cols, dtype=tl.int32)

    for block_id in tl.static_range(BLOCKS_PER_ROW):
        tile_offsets = (block_id * BLOCK_K + k_offsets).to(tl.int32)
        values = tl.load(
            x_ptr + row_bases + block_id * BLOCK_K + k_offsets,
            mask=row_mask[:, None],
            other=float("inf"),
        ).to(tl.float32)

        tile_min = tl.min(values, axis=1)
        tile_indices = tl.where(values == tile_min[:, None], tile_offsets,
                                invalid_index)
        tile_first_idx = tl.min(tile_indices, axis=1)

        should_update = (tile_min < best_val) | ((tile_min == best_val) &
                                                 (tile_first_idx < best_idx))
        best_val = tl.where(should_update, tile_min, best_val)
        best_idx = tl.where(should_update, tile_first_idx, best_idx)

    tl.store(out_ptr + row_offsets, best_idx, mask=row_mask)


@triton.jit
def _argmin_row_kernel_4096_fp32(
    x_ptr,
    out_ptr,
    rows,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < rows
    k_offsets = tl.arange(0, BLOCK_K)[None, :]
    row_bases = row_offsets[:, None] * ROW_STRIDE

    best_val = tl.full((BLOCK_M, ), float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_M, ), dtype=tl.int32)
    local_offsets = tl.arange(0, BLOCK_K)[None, :].to(tl.int32)
    invalid_local_index = tl.full((BLOCK_M, BLOCK_K), BLOCK_K, dtype=tl.int32)

    for block_id in tl.static_range(BLOCKS_PER_ROW):
        values = tl.load(
            x_ptr + row_bases + block_id * BLOCK_K + k_offsets,
            mask=row_mask[:, None],
            other=float("inf"),
        )

        tile_min = tl.min(values, axis=1)
        tile_indices = tl.where(values == tile_min[:, None], local_offsets,
                                invalid_local_index)
        tile_first_idx = tl.min(tile_indices, axis=1) + block_id * BLOCK_K

        should_update = (tile_min < best_val) | ((tile_min == best_val) &
                                                 (tile_first_idx < best_idx))
        best_val = tl.where(should_update, tile_min, best_val)
        best_idx = tl.where(should_update, tile_first_idx, best_idx)

    tl.store(out_ptr + row_offsets, best_idx, mask=row_mask)


def argmin_over_a_dimension(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("argmin_over_a_dimension expects a torch.Tensor input")
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "argmin_over_a_dimension expects an Ascend NPU tensor")
    if x.dim() == 0:
        raise ValueError(
            "argmin_over_a_dimension expects a tensor with rank at least 1")
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(
            "argmin_over_a_dimension supports only float16, float32, and bfloat16 inputs"
        )

    dim = int(dim)
    if dim < 0:
        dim += x.dim()
    if dim < 0 or dim >= x.dim():
        raise ValueError(
            f"invalid reduction dim {dim} for input rank {x.dim()}")
    if x.shape[dim] == 0:
        raise ValueError(
            "argmin_over_a_dimension does not support empty reduction axes")

    x_last = x.movedim(dim, -1).contiguous()
    rows = x_last.numel() // x_last.shape[-1]
    cols = x_last.shape[-1]
    x_2d = x_last.view(rows, cols)

    out = torch.empty((rows, ), device=x.device, dtype=torch.int32)
    block_k = 64
    while block_k < cols and block_k < 1024:
        block_k *= 2
    block_m = 8
    grid = (triton.cdiv(rows, block_m), )
    if cols == 4096 and block_k == 1024 and x_2d.dtype == torch.float32:
        _argmin_row_kernel_4096_fp32[grid](
            x_2d,
            out,
            rows,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCKS_PER_ROW=4,
            ROW_STRIDE=4096,
            num_warps=8,
            num_stages=2,
        )
    elif cols == 4096 and block_k == 1024:
        _argmin_row_kernel_4096[grid](
            x_2d,
            out,
            rows,
            cols,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCKS_PER_ROW=4,
            num_warps=8,
            num_stages=2,
        )
    elif cols % block_k == 0:
        _argmin_row_kernel_full_tiles[grid](
            x_2d,
            out,
            rows,
            cols,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_warps=8,
            num_stages=2,
        )
    else:
        _argmin_row_kernel[grid](
            x_2d,
            out,
            rows,
            cols,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_warps=8,
            num_stages=2,
        )

    out_shape = list(x.shape)
    del out_shape[dim]
    return out.view(*out_shape).to(torch.int64)


class ModelNew(nn.Module):
    """
    Argmin reduction over a specified dimension using Triton on Ascend NPU.
    """

    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmin_over_a_dimension(x, self.dim)
