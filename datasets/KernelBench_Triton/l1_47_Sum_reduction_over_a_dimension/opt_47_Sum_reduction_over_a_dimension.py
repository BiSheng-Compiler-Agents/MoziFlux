import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.runtime import driver

_MAX_GRID = 65535


def _num_vector_cores(device):
    try:
        props = driver.active.utils.get_device_properties(device)
        return int(props.get("num_vectorcore", props.get("num_aicore", 1)))
    except Exception:
        return 1


@triton.jit
def _sum_dim1_kernel(
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
    n_tiles = tl.cdiv(N, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, 16), BLOCK_N)

    for tile in tl.range(pid, total_tiles, n_programs):
        b = tile // n_tiles
        n_tile = tile - b * n_tiles
        n = n_tile * BLOCK_N + offs_n
        mask_n = n < N
        acc = tl.zeros((BLOCK_N, ), dtype=tl.float32)
        base = b * M * N

        for m0 in tl.range(0, M, BLOCK_M):
            m = m0 + offs_m
            ptrs = x_ptr + base + m[:, None] * N + n[None, :]
            mask = (m[:, None] < M) & mask_n[None, :]
            vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(vals, axis=0)

        tl.store(out_ptr + b * N + n, acc, mask=mask_n)


@triton.jit
def _sum_dim2_kernel(
    x_ptr,
    out_ptr,
    B: tl.constexpr,
    M,
    N: tl.constexpr,
    total_tiles,
    n_programs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    m_tiles = tl.cdiv(M, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, 16), BLOCK_N)

    for tile in tl.range(pid, total_tiles, n_programs):
        b = tile // m_tiles
        m_tile = tile - b * m_tiles
        m = m_tile * BLOCK_M + offs_m
        mask_m = m < M
        acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)
        base = b * M * N

        for n0 in tl.range(0, N, BLOCK_N):
            n = n0 + offs_n
            mask_n = n < N
            ptrs = x_ptr + base + m[:, None] * N + n[None, :]
            vals = tl.load(ptrs,
                           mask=mask_m[:, None] & mask_n[None, :],
                           other=0.0).to(tl.float32)
            acc += tl.sum(vals, axis=1)

        tl.store(out_ptr + b * M + m, acc, mask=mask_m)


@triton.jit
def _sum_dim0_kernel(
    x_ptr,
    out_ptr,
    B,
    M,
    N,
    total_tiles,
    n_programs,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    offs_b = tl.arange(0, BLOCK_B)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, 16), BLOCK_N)

    for tile in tl.range(pid, total_tiles, n_programs):
        m_tile = tile // n_tiles
        n_tile = tile - m_tile * n_tiles
        m = m_tile * BLOCK_M + offs_m
        n = n_tile * BLOCK_N + offs_n
        mask_m = m < M
        mask_n = n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for b0 in tl.range(0, B, BLOCK_B):
            b = b0 + offs_b
            mask_b = b < B
            ptrs = (x_ptr + b[:, None, None] * (M * N) + m[None, :, None] * N +
                    n[None, None, :])
            vals = tl.load(
                ptrs,
                mask=mask_b[:, None, None] & mask_m[None, :, None]
                & mask_n[None, None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(vals, axis=0)

        tl.store(out_ptr + m[:, None] * N + n[None, :],
                 acc,
                 mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    """Sum reduction over one dimension of a contiguous 3D NPU tensor."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected a 3D tensor, but got shape {tuple(x.shape)}.")
        if x.device.type != "npu":
            raise ValueError(
                f"Expected an NPU tensor, but got device {x.device}.")
        if not x.dtype.is_floating_point:
            raise TypeError(
                f"Expected a floating-point tensor, but got {x.dtype}.")

        dim = self.dim if self.dim >= 0 else x.dim() + self.dim
        if dim not in (0, 1, 2):
            raise ValueError(
                f"Expected reduction dim in [0, 1, 2], but got {self.dim}.")

        x = x if x.is_contiguous() else x.contiguous()
        B, M, N = x.shape
        device = torch.npu.current_device()
        cores = max(1, min(_num_vector_cores(device), _MAX_GRID))

        if dim == 1:
            out = torch.empty((B, 1, N), device=x.device, dtype=x.dtype)
            block_m = 128
            block_n = 128
            total_tiles = B * triton.cdiv(N, block_n)
            n_programs = max(1, min(cores, total_tiles, _MAX_GRID))
            _sum_dim1_kernel[(n_programs, )](
                x,
                out,
                B,
                M,
                N,
                total_tiles,
                n_programs,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=2,
            )
            return out

        if dim == 2:
            out = torch.empty((B, M, 1), device=x.device, dtype=x.dtype)
            block_m = 128
            block_n = 128
            total_tiles = B * triton.cdiv(M, block_m)
            n_programs = max(1, min(cores, total_tiles, _MAX_GRID))
            _sum_dim2_kernel[(n_programs, )](
                x,
                out,
                B,
                M,
                N,
                total_tiles,
                n_programs,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=2,
            )
            return out

        out = torch.empty((1, M, N), device=x.device, dtype=x.dtype)
        block_b = 8
        block_m = 16
        block_n = 64
        total_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
        n_programs = max(1, min(cores, total_tiles, _MAX_GRID))
        _sum_dim0_kernel[(n_programs, )](
            x,
            out,
            B,
            M,
            N,
            total_tiles,
            n_programs,
            BLOCK_B=block_b,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=2,
        )
        return out


batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1


def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device="npu")
    return [x]


def get_init_inputs():
    return [reduce_dim]
