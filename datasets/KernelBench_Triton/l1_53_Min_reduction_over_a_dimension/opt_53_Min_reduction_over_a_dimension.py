import torch
import torch.nn as nn

import triton
import triton.language as tl

_MAX_GRID = 65535


@triton.jit
def _min_reduce_dim1_tile_kernel(
    x_ptr,
    out_ptr,
    B,
    M,
    N,
    stride_b,
    stride_m,
    stride_n,
    out_stride_b,
    out_stride_n,
    total_tiles,
    n_tiles_n,
    n_programs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    while tile < total_tiles:
        b = tile // n_tiles_n
        n_blk = tile - b * n_tiles_n
        n = n_blk * BLOCK_N + offs_n
        n = tl.max_contiguous(tl.multiple_of(n, 16), BLOCK_N)
        n_mask = n < N

        acc = tl.full((BLOCK_N, ), float("inf"), dtype=tl.float32)
        m0 = 0
        base = b * stride_b
        while m0 < M:
            m = m0 + offs_m
            mask = (m[:, None] < M) & n_mask[None, :]
            vals = tl.load(
                x_ptr + base + m[:, None] * stride_m + n[None, :] * stride_n,
                mask=mask,
                other=float("inf"),
            )
            tile_min = tl.min(vals, axis=0)
            acc = tl.minimum(acc, tile_min)
            m0 += BLOCK_M

        tl.store(out_ptr + b * out_stride_b + n * out_stride_n,
                 acc,
                 mask=n_mask)
        tile += n_programs


@triton.jit
def _min_reduce_dim0_tile_kernel(
    x_ptr,
    out_ptr,
    B,
    M,
    N,
    stride_b,
    stride_m,
    stride_n,
    out_stride_m,
    out_stride_n,
    total_tiles,
    n_tiles_n,
    n_programs,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid
    offs_b = tl.arange(0, BLOCK_B)
    offs_n = tl.arange(0, BLOCK_N)
    while tile < total_tiles:
        m = tile // n_tiles_n
        n_blk = tile - m * n_tiles_n
        n = n_blk * BLOCK_N + offs_n
        n = tl.max_contiguous(tl.multiple_of(n, 16), BLOCK_N)
        n_mask = n < N

        acc = tl.full((BLOCK_N, ), float("inf"), dtype=tl.float32)
        b0 = 0
        base_m = m * stride_m
        while b0 < B:
            b = b0 + offs_b
            mask = (b[:, None] < B) & n_mask[None, :]
            vals = tl.load(
                x_ptr + b[:, None] * stride_b + base_m + n[None, :] * stride_n,
                mask=mask,
                other=float("inf"),
            )
            tile_min = tl.min(vals, axis=0)
            acc = tl.minimum(acc, tile_min)
            b0 += BLOCK_B

        tl.store(out_ptr + m * out_stride_m + n * out_stride_n,
                 acc,
                 mask=n_mask)
        tile += n_programs


@triton.jit
def _min_reduce_dim2_row_kernel(
    x_ptr,
    out_ptr,
    B,
    M,
    N,
    stride_b,
    stride_m,
    stride_n,
    out_stride_b,
    out_stride_m,
    total_rows,
    n_programs,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    offs_n = tl.arange(0, BLOCK_N)
    while row < total_rows:
        b = row // M
        m = row - b * M
        acc = tl.full((BLOCK_N, ), float("inf"), dtype=tl.float32)
        n0 = 0
        base = b * stride_b + m * stride_m
        while n0 < N:
            n = n0 + offs_n
            n = tl.max_contiguous(tl.multiple_of(n, 16), BLOCK_N)
            mask = n < N
            vals = tl.load(x_ptr + base + n * stride_n,
                           mask=mask,
                           other=float("inf"))
            acc = tl.minimum(acc, vals)
            n0 += BLOCK_N
        out = tl.min(acc, axis=0)
        tl.store(out_ptr + b * out_stride_b + m * out_stride_m, out)
        row += n_programs


class ModelNew(nn.Module):
    """Min reduction over one dimension of a 3D tensor with Ascend Triton kernels."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @staticmethod
    def _block_m(k: int) -> int:
        return 128 if k >= 128 else 64

    @staticmethod
    def _block_n(n: int) -> int:
        return 128 if n >= 128 else 64 if n >= 64 else 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"ModelNew expects a 3D tensor, got shape {tuple(x.shape)}")
        if not hasattr(torch, "npu") or x.device.type != "npu":
            raise ValueError("ModelNew requires an Ascend NPU tensor input")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                f"Unsupported dtype for Triton reduction: {x.dtype}")

        dim = self.dim
        if dim < 0:
            dim += 3
        if dim not in (0, 1, 2):
            raise ValueError(
                f"Unsupported reduction dim {self.dim} for 3D input")

        B, M, N = x.shape
        if B == 0 or M == 0 or N == 0:
            raise ValueError("Zero-sized reductions are not supported")
        sb, sm, sn = x.stride()

        if dim == 1:
            out = torch.empty((B, N), device=x.device, dtype=x.dtype)
            ob, on = out.stride()
            block_m = self._block_m(M)
            block_n = self._block_n(N)
            n_tiles_n = triton.cdiv(N, block_n)
            total_tiles = B * n_tiles_n
            n_programs = min(total_tiles, _MAX_GRID)
            _min_reduce_dim1_tile_kernel[(n_programs, )](
                x,
                out,
                B,
                M,
                N,
                sb,
                sm,
                sn,
                ob,
                on,
                total_tiles,
                n_tiles_n,
                n_programs,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=2,
            )
            return out

        if dim == 0:
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            om, on = out.stride()
            block_b = self._block_m(B)
            block_n = self._block_n(N)
            n_tiles_n = triton.cdiv(N, block_n)
            total_tiles = M * n_tiles_n
            n_programs = min(total_tiles, _MAX_GRID)
            _min_reduce_dim0_tile_kernel[(n_programs, )](
                x,
                out,
                B,
                M,
                N,
                sb,
                sm,
                sn,
                om,
                on,
                total_tiles,
                n_tiles_n,
                n_programs,
                BLOCK_B=block_b,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=2,
            )
            return out

        out = torch.empty((B, M), device=x.device, dtype=x.dtype)
        ob, om = out.stride()
        block_n = 256 if N >= 256 else self._block_n(N)
        total_rows = B * M
        n_programs = min(total_rows, _MAX_GRID)
        _min_reduce_dim2_row_kernel[(n_programs, )](
            x,
            out,
            B,
            M,
            N,
            sb,
            sm,
            sn,
            ob,
            om,
            total_rows,
            n_programs,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=2,
        )
        return out


def min_reduce_triton(x: torch.Tensor, dim: int) -> torch.Tensor:
    return ModelNew(dim)(x)
