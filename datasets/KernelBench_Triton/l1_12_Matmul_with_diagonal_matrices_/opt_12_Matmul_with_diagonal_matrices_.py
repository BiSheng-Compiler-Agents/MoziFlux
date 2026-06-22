"""
Optimized row-scaling kernel: C[i,j] = A[i] * B[i,j]

Baseline trace showed SCALARLDST bottleneck at 97.7% (2435/2493 wall cycles):
- ST_XD_XN_IMM: 1263 cy (scalar spill from if/else branch)
- LD_XD_XN_IMM: 1044 cy (args loading)
- LDP_XI_XJ_XN: 2546 cy over 5 events (data loading)
- Actual compute (RVECEX): only 13 cy

Optimizations applied:
1. Removed if/else branch → single masked path, eliminates scalar control-flow spill
2. Removed cache_modifier=".cg" (CUDA-only, silent compile killer on Ascend)
3. Two-path dispatch: direct (1D grid ≤ 65535) + persistent (work-stealing > 65535)
4. Larger BLOCK_N (64 → 1024): better throughput, fewer programs
5. Simplified pointer math with tl.max_contiguous / tl.multiple_of hints
"""

import torch
import triton
import triton.language as tl

MAX_PROGRAMS = 65535  # Ascend FFTS grid cap


# ──────────────────────────────────────────────────────────────────────────────
# Direct path: 1D grid, one program per (row, col_block) tile
# Used when total tiles ≤ 65535 — no while-loop overhead
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _row_scale_direct(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    M,
    stride_bm,
    stride_bn,
    stride_cm,
    stride_cn,
    num_col_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    # 1D → 2D remap: row and col_block
    row = pid // num_col_blocks
    col_block = pid % num_col_blocks
    col_start = col_block * BLOCK_N

    offs_n = col_start + tl.arange(0, BLOCK_N)
    tl.max_contiguous(offs_n, BLOCK_N)
    tl.multiple_of(offs_n, 16)

    # Single combined mask — no if/else branch
    mask = (row < N) & (offs_n < M)

    # Load A scalar (row value, broadcast across columns)
    a_val = tl.load(a_ptr + row, mask=row < N, other=0.0)

    # Load B tile — contiguous row segment
    b_ptrs = b_ptr + row * stride_bm + offs_n * stride_bn
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    # Scale and store
    c_ptrs = c_ptr + row * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, b * a_val, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Persistent path: work-stealing loop, caps grid at MAX_PROGRAMS
# Used when total tiles > 65535 — avoids FFTS crash
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _row_scale_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    M,
    stride_bm,
    stride_bn,
    stride_cm,
    stride_cn,
    total_tiles,
    num_col_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for tile_idx in range(pid, total_tiles, num_programs):
        row = tile_idx // num_col_blocks
        col_block = tile_idx % num_col_blocks
        col_start = col_block * BLOCK_N

        offs_n = col_start + tl.arange(0, BLOCK_N)
        tl.max_contiguous(offs_n, BLOCK_N)
        tl.multiple_of(offs_n, 16)

        mask = (row < N) & (offs_n < M)

        a_val = tl.load(a_ptr + row, mask=row < N, other=0.0)
        b_ptrs = b_ptr + row * stride_bm + offs_n * stride_bn
        b = tl.load(b_ptrs, mask=mask, other=0.0)
        c_ptrs = c_ptr + row * stride_cm + offs_n * stride_cn
        tl.store(c_ptrs, b * a_val, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Host interface
# ──────────────────────────────────────────────────────────────────────────────
class ModelNew(torch.nn.Module):
    """Row-scale: C[i,j] = A[i] * B[i,j] for diagonal matrix multiplication."""

    def __init__(self, BLOCK_N: int = 1024):
        super().__init__()
        assert BLOCK_N % 16 == 0, "BLOCK_N must be multiple of 16"
        self.BLOCK_N = BLOCK_N

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: diagonal vector, shape [N], dtype fp16
            b: matrix, shape [N, M], dtype fp16
        Returns:
            c: output, shape [N, M], dtype fp16
        """
        assert a.device == b.device, "a and b must be on same device"
        assert a.dtype == b.dtype == torch.float16, "Both inputs must be float16"
        assert a.dim() == 1 and b.dim() == 2, "a=[N], b=[N,M]"
        assert a.shape[0] == b.shape[0], "a and b must have same N"

        N, M = a.shape[0], b.shape[1]
        BLOCK_N = self.BLOCK_N
        num_col_blocks = triton.cdiv(M, BLOCK_N)
        total_tiles = N * num_col_blocks

        c = torch.empty(N, M, device=a.device, dtype=torch.float16)

        if total_tiles > MAX_PROGRAMS:
            # Persistent path — cap grid, each program processes multiple tiles
            n_programs = min(total_tiles, MAX_PROGRAMS)
            grid = (n_programs, )
            _row_scale_persistent[grid](
                a,
                b,
                c,
                N,
                M,
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                total_tiles,
                num_col_blocks=num_col_blocks,
                BLOCK_N=BLOCK_N,
            )
        else:
            # Direct path — one program per tile, no while-loop overhead
            grid = (total_tiles, )
            _row_scale_direct[grid](
                a,
                b,
                c,
                N,
                M,
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                num_col_blocks=num_col_blocks,
                BLOCK_N=BLOCK_N,
            )

        return c
