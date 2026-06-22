import torch
import torch.nn as nn
import triton
import triton.language as tl

DIM1_BLOCK_M = 32
DIM1_BLOCK_N = 256
DIM1_NUM_WARPS = 8
DIM1_NUM_STAGES = 4


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _max_reduce_dim1_kernel(x_ptr, o_ptr, B, M, N, sx0, sx1, sx2, so0, so1,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Grid: (B, ceil_div(N, BLOCK_N))
    b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Hints for vectorization along N
    tl.multiple_of(n_offsets, 16)
    tl.max_contiguous(n_offsets, BLOCK_N)

    base = x_ptr + b * sx0
    col_offsets = tl.max_contiguous(tl.multiple_of(n_offsets, 16), BLOCK_N)
    col_ptrs = base + col_offsets[None, :] * sx2
    m_offsets = tl.arange(0, BLOCK_M)[:, None]
    col_mask = n_mask[None, :]

    # Initialize accumulator without touching output memory.
    o_ptrs = o_ptr + b * so0 + n_offsets * so1
    acc = tl.full([BLOCK_N], -float("inf"), tl.float32)

    m_start = 0
    while m_start < M:
        row_offsets = m_start + m_offsets
        row_mask = row_offsets < M
        ptrs = col_ptrs + row_offsets * sx1
        x_tile = tl.load(
            ptrs,
            mask=row_mask & col_mask,
            other=-float("inf"),
            cache_modifier=".cg",
        )
        tile_max = tl.max(x_tile, axis=0).to(tl.float32)
        acc = tl.maximum(acc, tile_max)
        m_start += BLOCK_M

    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def _max_reduce_dim0_kernel(x_ptr, o_ptr, B, M, N, sx0, sx1, sx2, so0, so1,
                            BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr):
    # Grid: (M, ceil_div(N, BLOCK_N))
    m = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    o_ptrs = o_ptr + m * so0 + n_offsets * so1
    acc = tl.full([BLOCK_N], -float("inf"), tl.float32)

    b_start = 0
    while b_start < B:
        b_offsets = b_start + tl.arange(0, BLOCK_B)
        b_mask = b_offsets < B
        ptrs = x_ptr + b_offsets[:, None] * sx0 + m * sx1 + n_offsets[
            None, :] * sx2
        mask = b_mask[:, None] & n_mask[None, :]
        x_tile = tl.load(ptrs, mask=mask, other=-float("inf"))
        tile_max = tl.max(x_tile, axis=0).to(tl.float32)
        acc = tl.maximum(acc, tile_max)
        b_start += BLOCK_B

    tl.store(o_ptrs, acc, mask=n_mask)


@triton.jit
def _max_reduce_dim2_kernel(x_ptr, o_ptr, B, M, N, sx0, sx1, sx2, so0, so1,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Grid: (B, ceil_div(M, BLOCK_M))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # Single-buffered streaming across N
    n_start = 0
    # Initialize accumulator without touching output memory.
    o_ptrs = o_ptr + b * so0 + m_offsets * so1
    acc = tl.full([BLOCK_M], -float("inf"), tl.float32)

    while n_start < N:
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        ptrs = x_ptr + b * sx0 + m_offsets[:, None] * sx1 + n_offsets[
            None, :] * sx2
        mask = m_mask[:, None] & n_mask[None, :]
        x_tile = tl.load(ptrs,
                         mask=mask,
                         other=-float("inf"),
                         cache_modifier=".cg")

        tile_max = tl.max(x_tile, axis=1).to(tl.float32)
        acc = tl.maximum(acc, tile_max)
        n_start += BLOCK_N

    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=m_mask)


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
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

    if dim == 1:
        out = torch.empty((B, N), device=x.device, dtype=x.dtype)
        so0, so1 = out.stride()
        grid = (B, triton.cdiv(N, DIM1_BLOCK_N))
        _max_reduce_dim1_kernel[grid](x_c,
                                      out,
                                      B,
                                      M,
                                      N,
                                      sx0,
                                      sx1,
                                      sx2,
                                      so0,
                                      so1,
                                      BLOCK_M=DIM1_BLOCK_M,
                                      BLOCK_N=DIM1_BLOCK_N,
                                      num_warps=DIM1_NUM_WARPS,
                                      num_stages=DIM1_NUM_STAGES)
        return out

    if dim == 0:
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        so0, so1 = out.stride()
        BLOCK_B, BLOCK_N = 16, 128
        grid = (M, triton.cdiv(N, BLOCK_N))
        _max_reduce_dim0_kernel[grid](x_c,
                                      out,
                                      B,
                                      M,
                                      N,
                                      sx0,
                                      sx1,
                                      sx2,
                                      so0,
                                      so1,
                                      BLOCK_B=BLOCK_B,
                                      BLOCK_N=BLOCK_N,
                                      num_warps=4,
                                      num_stages=4)
        return out

    out = torch.empty((B, M), device=x.device, dtype=x.dtype)
    so0, so1 = out.stride()
    BLOCK_M, BLOCK_N = 128, 128
    grid = (B, triton.cdiv(M, BLOCK_M))
    _max_reduce_dim2_kernel[grid](x_c,
                                  out,
                                  B,
                                  M,
                                  N,
                                  sx0,
                                  sx1,
                                  sx2,
                                  so0,
                                  so1,
                                  BLOCK_M=BLOCK_M,
                                  BLOCK_N=BLOCK_N,
                                  num_warps=8,
                                  num_stages=4)
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
    return [1]  # Example, change to desired dimension
