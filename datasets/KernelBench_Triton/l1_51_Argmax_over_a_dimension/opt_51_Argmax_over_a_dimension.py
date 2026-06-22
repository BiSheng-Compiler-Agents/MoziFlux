import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_MAX_GRID = 65535
_BLOCK_M = 64
_BLOCK_N = 128


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _argmax_dim1_tile_kernel(
    x_ptr,
    out_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    total_tiles: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    n_tiles = tl.cdiv(N, BLOCK_N)

    for tile_id in tl.range(pid, total_tiles, n_programs):
        b = tile_id // n_tiles
        n_tile = tile_id - b * n_tiles
        offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.multiple_of(offs_n, 16)
        tl.max_contiguous(offs_n, BLOCK_N)
        n_mask = offs_n < N

        best_val = tl.full((BLOCK_N, ), -float("inf"), dtype=tl.float32)
        best_idx = tl.zeros((BLOCK_N, ), dtype=tl.int32)

        for m0 in tl.range(0, M, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            ptrs = x_ptr + b * M * N + offs_m[:, None] * N + offs_n[None, :]
            mask = (offs_m[:, None] < M) & n_mask[None, :]
            vals = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
            tile_max = tl.max(vals, axis=0)

            eq = (vals == tile_max[None, :]) & mask
            invalid = tl.full((BLOCK_M, BLOCK_N), M, dtype=tl.int32)
            idxs = tl.where(eq, offs_m[:, None].to(tl.int32), invalid)
            tile_idx = tl.min(idxs, axis=0)

            update = (tile_max > best_val) | ((tile_max == best_val) &
                                              (tile_idx < best_idx))
            best_val = tl.where(update, tile_max, best_val)
            best_idx = tl.where(update, tile_idx, best_idx)

        tl.store(out_ptr + b * N + offs_n, best_idx.to(tl.int64), mask=n_mask)


def _argmax_dim1_triton(x: torch.Tensor) -> torch.Tensor:
    B, M, N = x.shape
    out = torch.empty((B, N), device=x.device, dtype=torch.int64)
    total_tiles = B * triton.cdiv(N, _BLOCK_N)
    grid = (min(total_tiles, _MAX_GRID), )
    _argmax_dim1_tile_kernel[grid](
        x,
        out,
        B,
        M,
        N,
        total_tiles,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out


def argmax_over_a_dimension(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("argmax_over_a_dimension expects a torch.Tensor input")
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "argmax_over_a_dimension expects an Ascend NPU tensor")
    if x.dim() == 0:
        raise ValueError(
            "argmax_over_a_dimension expects a tensor with rank at least 1")
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(
            "argmax_over_a_dimension supports only float16, float32, and bfloat16 inputs"
        )

    dim = int(dim)
    if dim < 0:
        dim += x.dim()
    if dim < 0 or dim >= x.dim():
        raise ValueError(
            f"invalid reduction dim {dim} for input rank {x.dim()}")
    if x.shape[dim] == 0:
        raise ValueError(
            "argmax_over_a_dimension does not support empty reduction axes")

    if x.dim() == 3 and dim == 1 and x.is_contiguous():
        return _argmax_dim1_triton(x)
    return torch.argmax(x, dim=dim)


class ModelNew(nn.Module):
    """Argmax reduction over a specified dimension using Triton on Ascend NPU."""

    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmax_over_a_dimension(x, self.dim)
