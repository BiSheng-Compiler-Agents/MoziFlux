import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.runtime import driver

_MAX_GRID = 65535
_DIM1_BLOCK_M = 64
_DIM1_BLOCK_N = 64
_LAST_BLOCK_K = 1024
_DIM0_BLOCK_N = 64


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


def _num_vector_cores(x: torch.Tensor) -> int:
    try:
        props = driver.active.utils.get_device_properties(x.device)
        return int(props.get("num_vectorcore", props.get("num_aicore", 20)))
    except Exception:
        return 20


@triton.jit
def _argmin_dim1_kernel(
    x_ptr,
    out_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    N,
    total_tiles,
    n_programs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid
    n_blocks = tl.cdiv(N, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    while tile < total_tiles:
        b = tile // n_blocks
        nb = tile - b * n_blocks
        n0 = nb * BLOCK_N
        n_idxs = n0 + offs_n
        n_mask = n_idxs < N

        best_val = tl.full((BLOCK_N, ), float("inf"), dtype=tl.float32)
        best_idx = tl.full((BLOCK_N, ), M, dtype=tl.int32)

        m0 = 0
        while m0 < M:
            m_idxs = m0 + offs_m
            mask = (m_idxs[:, None] < M) & n_mask[None, :]
            vals = tl.load(
                x_ptr + b * M * N + m_idxs[:, None] * N + n_idxs[None, :],
                mask=mask,
                other=float("inf"),
            ).to(tl.float32)
            tile_min = tl.min(vals, axis=0)
            is_first_min = (vals == tile_min[None, :]) & mask
            idx_mat = tl.where(is_first_min, m_idxs[:, None], M)
            tile_idx = tl.min(idx_mat, axis=0)
            update = (tile_min < best_val) | ((tile_min == best_val) &
                                              (tile_idx < best_idx))
            best_val = tl.where(update, tile_min, best_val)
            best_idx = tl.where(update, tile_idx, best_idx)
            m0 += BLOCK_M

        tl.store(out_ptr + b * N + n_idxs, best_idx.to(tl.int64), mask=n_mask)
        tile += n_programs


@triton.jit
def _argmin_lastdim_kernel(
    x_ptr,
    out_ptr,
    rows,
    cols,
    n_programs,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    offs = tl.arange(0, BLOCK_K)
    while row < rows:
        base = row * cols
        best_val = tl.full((), float("inf"), dtype=tl.float32)
        best_idx = tl.full((), 0, dtype=tl.int32)
        k0 = 0
        while k0 < cols:
            k = k0 + offs
            mask = k < cols
            vals = tl.load(x_ptr + base + k, mask=mask,
                           other=float("inf")).to(tl.float32)
            tile_min = tl.min(vals, axis=0)
            eq = (vals == tile_min) & mask
            idxs = tl.where(eq, k, cols)
            tile_idx = tl.min(idxs, axis=0)
            update = (tile_min < best_val) | ((tile_min == best_val) &
                                              (tile_idx < best_idx))
            best_val = tl.where(update, tile_min, best_val)
            best_idx = tl.where(update, tile_idx, best_idx)
            k0 += BLOCK_K
        tl.store(out_ptr + row, best_idx.to(tl.int64))
        row += n_programs


@triton.jit
def _argmin_dim0_3d_kernel(
    x_ptr,
    out_ptr,
    B,
    M,
    N,
    total_tiles,
    n_programs,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid
    offs_b = tl.arange(0, BLOCK_B)
    offs_n = tl.arange(0, BLOCK_N)
    while tile < total_tiles:
        m = tile // tl.cdiv(N, BLOCK_N)
        nb = tile - m * tl.cdiv(N, BLOCK_N)
        n = nb * BLOCK_N + offs_n
        n_mask = n < N
        best_val = tl.full((BLOCK_N, ), float("inf"), dtype=tl.float32)
        best_idx = tl.full((BLOCK_N, ), 0, dtype=tl.int32)
        b0 = 0
        while b0 < B:
            b = b0 + offs_b
            vals = tl.load(
                x_ptr + b[:, None] * M * N + m * N + n[None, :],
                mask=(b[:, None] < B) & n_mask[None, :],
                other=float("inf"),
            ).to(tl.float32)
            min_val = tl.min(vals, axis=0)
            eq = (vals == min_val[None, :]) & (b[:, None]
                                               < B) & n_mask[None, :]
            idx_mat = tl.where(eq, b[:, None], B)
            idx = tl.min(idx_mat, axis=0)
            update = (min_val < best_val) | ((min_val == best_val) &
                                             (idx < best_idx))
            best_val = tl.where(update, min_val, best_val)
            best_idx = tl.where(update, idx, best_idx)
            b0 += BLOCK_B
        tl.store(out_ptr + m * N + n, best_idx.to(tl.int64), mask=n_mask)
        tile += n_programs


def _launch_dim1_3d(x: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    B, M, N = x_c.shape
    out = torch.empty((B, N), device=x.device, dtype=torch.int64)
    total_tiles = B * triton.cdiv(N, _DIM1_BLOCK_N)
    nprog = max(1, min(_MAX_GRID, _num_vector_cores(x), total_tiles))
    _argmin_dim1_kernel[(nprog, )](
        x_c,
        out,
        B,
        M,
        N,
        total_tiles,
        nprog,
        BLOCK_M=_DIM1_BLOCK_M,
        BLOCK_N=_DIM1_BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out


def _launch_dim0_3d(x: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    B, M, N = x_c.shape
    out = torch.empty((M, N), device=x.device, dtype=torch.int64)
    total_tiles = M * triton.cdiv(N, _DIM0_BLOCK_N)
    nprog = max(1, min(_MAX_GRID, _num_vector_cores(x), total_tiles))
    _argmin_dim0_3d_kernel[(nprog, )](
        x_c,
        out,
        B,
        M,
        N,
        total_tiles,
        nprog,
        BLOCK_B=64,
        BLOCK_N=_DIM0_BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out


def _launch_lastdim(x_last: torch.Tensor, out_shape) -> torch.Tensor:
    x_c = x_last.contiguous()
    cols = x_c.shape[-1]
    rows = x_c.numel() // cols
    out = torch.empty((rows, ), device=x_c.device, dtype=torch.int64)
    block_k = 64
    while block_k < cols and block_k < _LAST_BLOCK_K:
        block_k *= 2
    nprog = max(1, min(_MAX_GRID, _num_vector_cores(x_c), rows))
    _argmin_lastdim_kernel[(nprog, )](
        x_c.view(rows, cols),
        out,
        rows,
        cols,
        nprog,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return out.view(*out_shape)


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

    if x.dim() == 3 and dim == 1:
        return _launch_dim1_3d(x)
    if x.dim() == 3 and dim == 0:
        return _launch_dim0_3d(x)

    out_shape = list(x.shape)
    del out_shape[dim]
    if dim == x.dim() - 1:
        return _launch_lastdim(x, out_shape)
    return _launch_lastdim(x.movedim(dim, -1), out_shape)


class ModelNew(nn.Module):
    """Argmin reduction over a specified dimension using Triton on Ascend NPU."""

    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmin_over_a_dimension(x, self.dim)


batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1


def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    return [dim]
