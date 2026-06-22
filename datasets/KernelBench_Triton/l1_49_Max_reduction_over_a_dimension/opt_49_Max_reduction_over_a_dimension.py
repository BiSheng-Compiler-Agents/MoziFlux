import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver

_MAX_GRID = 65535


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


def _num_vector_cores(device) -> int:
    try:
        props = driver.active.utils.get_device_properties(device)
        return int(
            props.get("num_vectorcore", props.get("num_aicore", _MAX_GRID)))
    except Exception:
        return _MAX_GRID


@triton.jit
def _max_reduce_dim1_kernel(
    x_ptr,
    o_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    sx2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    total_tiles,
    n_programs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks: tl.constexpr = tl.cdiv(N, BLOCK_N)

    for tile_id in tl.range(pid, total_tiles, n_programs):
        b = tile_id // n_blocks
        nb = tile_id - b * n_blocks
        n_offsets = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        tl.multiple_of(n_offsets, 16)
        tl.max_contiguous(n_offsets, BLOCK_N)

        acc = tl.full([BLOCK_N], -float("inf"), tl.float32)
        m_offsets_base = tl.arange(0, BLOCK_M)
        for m_start in tl.range(0, M, BLOCK_M):
            m_offsets = m_start + m_offsets_base
            ptrs = x_ptr + b * sx0 + m_offsets[:, None] * sx1 + n_offsets[
                None, :] * sx2
            mask = (m_offsets[:, None] < M) & n_mask[None, :]
            vals = tl.load(ptrs,
                           mask=mask,
                           other=-float("inf"),
                           cache_modifier=".cg").to(tl.float32)
            tile_max = tl.max(vals, axis=0)
            acc = tl.maximum(acc, tile_max)

        tl.store(o_ptr + b * so0 + n_offsets * so1,
                 acc.to(o_ptr.dtype.element_ty),
                 mask=n_mask)


@triton.jit
def _max_reduce_dim0_kernel(
    x_ptr,
    o_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    sx2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    total_tiles,
    n_programs,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks: tl.constexpr = tl.cdiv(N, BLOCK_N)

    for tile_id in tl.range(pid, total_tiles, n_programs):
        m = tile_id // n_blocks
        nb = tile_id - m * n_blocks
        n_offsets = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        acc = tl.full([BLOCK_N], -float("inf"), tl.float32)
        b_base = tl.arange(0, BLOCK_B)

        for b_start in tl.range(0, B, BLOCK_B):
            b_offsets = b_start + b_base
            ptrs = x_ptr + b_offsets[:, None] * sx0 + m * sx1 + n_offsets[
                None, :] * sx2
            mask = (b_offsets[:, None] < B) & n_mask[None, :]
            vals = tl.load(ptrs,
                           mask=mask,
                           other=-float("inf"),
                           cache_modifier=".cg").to(tl.float32)
            tile_max = tl.max(vals, axis=0)
            acc = tl.maximum(acc, tile_max)

        tl.store(o_ptr + m * so0 + n_offsets * so1,
                 acc.to(o_ptr.dtype.element_ty),
                 mask=n_mask)


@triton.jit
def _max_reduce_dim2_kernel(
    x_ptr,
    o_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    sx2: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    total_tiles,
    n_programs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    m_blocks: tl.constexpr = tl.cdiv(M, BLOCK_M)

    for tile_id in tl.range(pid, total_tiles, n_programs):
        b = tile_id // m_blocks
        mb = tile_id - b * m_blocks
        m_offsets = mb * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        acc = tl.full([BLOCK_M], -float("inf"), tl.float32)
        n_base = tl.arange(0, BLOCK_N)

        for n_start in tl.range(0, N, BLOCK_N):
            n_offsets = n_start + n_base
            ptrs = x_ptr + b * sx0 + m_offsets[:, None] * sx1 + n_offsets[
                None, :] * sx2
            mask = m_mask[:, None] & (n_offsets[None, :] < N)
            vals = tl.load(ptrs,
                           mask=mask,
                           other=-float("inf"),
                           cache_modifier=".cg").to(tl.float32)
            tile_max = tl.max(vals, axis=1)
            acc = tl.maximum(acc, tile_max)

        tl.store(o_ptr + b * so0 + m_offsets * so1,
                 acc.to(o_ptr.dtype.element_ty),
                 mask=m_mask)


class ModelNew(nn.Module):
    """Max reduction over one dimension of a 3D tensor."""

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return max_reduction_over_a_dimension(x, self.dim)


def max_reduction_over_a_dimension(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "max_reduction_over_a_dimension expects an Ascend NPU tensor")
    if x.dim() != 3:
        raise ValueError(
            f"max_reduction_over_a_dimension expects a 3D tensor, got {x.dim()}D"
        )
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(f"unsupported dtype for max reduction: {x.dtype}")

    B, M, N = x.shape
    dim = dim if dim >= 0 else x.dim() + dim
    if dim not in (0, 1, 2):
        raise ValueError(
            f"reduction dim must be one of 0, 1, 2 or a negative alias, got {dim}"
        )

    x_c = x.contiguous()
    sx0, sx1, sx2 = x_c.stride()
    vec_cores = _num_vector_cores(x.device)

    if dim == 1:
        out = torch.empty((B, N), device=x.device, dtype=x.dtype)
        so0, so1 = out.stride()
        BLOCK_M, BLOCK_N = 32, 128
        total_tiles = B * triton.cdiv(N, BLOCK_N)
        n_programs = min(total_tiles, vec_cores, _MAX_GRID)
        _max_reduce_dim1_kernel[(n_programs, )](
            x_c,
            out,
            B,
            M,
            N,
            sx0,
            sx1,
            sx2,
            so0,
            so1,
            total_tiles,
            n_programs,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return out

    if dim == 0:
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        so0, so1 = out.stride()
        BLOCK_B, BLOCK_N = 32, 128
        total_tiles = M * triton.cdiv(N, BLOCK_N)
        n_programs = min(total_tiles, vec_cores, _MAX_GRID)
        _max_reduce_dim0_kernel[(n_programs, )](
            x_c,
            out,
            B,
            M,
            N,
            sx0,
            sx1,
            sx2,
            so0,
            so1,
            total_tiles,
            n_programs,
            BLOCK_B=BLOCK_B,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return out

    out = torch.empty((B, M), device=x.device, dtype=x.dtype)
    so0, so1 = out.stride()
    BLOCK_M, BLOCK_N = 128, 128
    total_tiles = B * triton.cdiv(M, BLOCK_M)
    n_programs = min(total_tiles, vec_cores, _MAX_GRID)
    _max_reduce_dim2_kernel[(n_programs, )](
        x_c,
        out,
        B,
        M,
        N,
        sx0,
        sx1,
        sx2,
        so0,
        so1,
        total_tiles,
        n_programs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out


batch_size = 128
dim1 = 4096
dim2 = 4095


def get_inputs():
    device = "npu" if hasattr(torch,
                              "npu") and torch.npu.is_available() else "cpu"
    x = torch.rand(batch_size, dim1, dim2, device=device)
    return [x]


def get_init_inputs():
    return [1]
