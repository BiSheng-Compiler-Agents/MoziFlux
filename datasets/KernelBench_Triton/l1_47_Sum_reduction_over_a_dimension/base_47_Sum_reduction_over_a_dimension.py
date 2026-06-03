import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _reduce_dim1_kernel(
    x_ptr, out_ptr, B, M, N,  # reduce over dim=1 (M)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # PIDs: (b, n-tile)
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    tl.max_contiguous(n_offsets, BLOCK_N)
    tl.multiple_of(n_offsets, 8)

    # Accumulator across M for a vector of N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    base_b = pid_b * M * N

    for m_start in range(0, M, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        ptrs = x_ptr + base_b + m_offsets[:, None] * N + n_offsets[None, :]
        if (m_start + BLOCK_M <= M) and (n_start + BLOCK_N <= N):
            vals = tl.load(ptrs)
        else:
            vals = tl.load(
                ptrs,
                mask=(m_offsets[:, None] < M) & mask_n[None, :],
                other=0.0,
            )
        acc += tl.sum(vals.to(tl.float32), axis=0)

    # store to out: shape [B, 1, N] contiguous -> offset b*N + n
    out_ptrs = out_ptr + pid_b * N + n_offsets
    tl.store(out_ptrs, acc, mask=mask_n)


@triton.jit
def _reduce_dim2_kernel(
    x_ptr, out_ptr, B, M, N,  # reduce over dim=2 (N)
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_m_tile = tl.program_id(1)

    m_start = pid_m_tile * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    base_b = pid_b * M * N
    # Stream over contiguous N in tiles for coalesced loads
    for k in range(0, tl.cdiv(N, BLOCK_K)):
        k_offsets = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < N
        ptrs = x_ptr + base_b + m_offsets[:, None] * N + k_offsets[None, :]
        # Use maskless fast path when tile fully inside bounds
        if (m_start + BLOCK_M <= M) and (k * BLOCK_K + BLOCK_K <= N):
            x = tl.load(ptrs)
        else:
            x = tl.load(ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        x = x.to(tl.float32)
        acc += tl.sum(x, axis=1)

    # store to out: shape [B, M, 1] contiguous -> offset b*M + m
    out_ptrs = out_ptr + pid_b * M + m_offsets
    tl.store(out_ptrs, acc, mask=mask_m)


@triton.jit
def _reduce_dim0_kernel(
    x_ptr, out_ptr, B, M, N,  # reduce over dim=0 (B)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr
):
    pid_m_tile = tl.program_id(0)
    pid_n_tile = tl.program_id(1)

    m_start = pid_m_tile * BLOCK_M
    n_start = pid_n_tile * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # loop over B in chunks (streaming to keep register usage low)
    for kb in range(0, tl.cdiv(B, BLOCK_KB)):
        b_offsets = kb * BLOCK_KB + tl.arange(0, BLOCK_KB)
        mask_b = b_offsets < B
        ptrs = (
            x_ptr
            + b_offsets[:, None, None] * (M * N)
            + m_offsets[None, :, None] * N
            + n_offsets[None, None, :]
        )
        # Use maskless fast path when tile fully inside bounds
        if (kb * BLOCK_KB + BLOCK_KB <= B) and (m_start + BLOCK_M <= M) and (n_start + BLOCK_N <= N):
            x = tl.load(ptrs)
        else:
            x = tl.load(ptrs, mask=mask_b[:, None, None] & mask_m[None, :, None] & mask_n[None, None, :], other=0.0)
        x = x.to(tl.float32)
        acc += tl.sum(x, axis=0)

    # store to out: shape [1, M, N] contiguous -> offset m*N + n
    out_ptrs = out_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    """
    Sum reduction over a specified dimension using Triton kernels on Ascend NPU.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D tensor, but got shape {tuple(x.shape)}.")
        if x.device.type != "npu":
            raise ValueError(f"Expected an NPU tensor, but got device {x.device}.")
        x = x.contiguous()
        B, M, N = x.shape
        dim = self.dim if self.dim >= 0 else x.dim() + self.dim

        if not x.dtype.is_floating_point:
            raise TypeError(f"Expected a floating-point tensor, but got {x.dtype}.")
        if dim not in (0, 1, 2):
            raise ValueError(f"Expected reduction dim in [0, 1, 2], but got {self.dim}.")

        if dim == 1:
            out = torch.empty((B, 1, N), device=x.device, dtype=x.dtype)
            grid = lambda META: (B, triton.cdiv(N, META['BLOCK_N']))
            _reduce_dim1_kernel[grid](x, out, B, M, N, BLOCK_M=40, BLOCK_N=128, num_warps=8, num_stages=2)
            return out
        if dim == 2:
            out = torch.empty((B, M, 1), device=x.device, dtype=x.dtype)
            grid = lambda META: (B, triton.cdiv(M, META['BLOCK_M']))
            _reduce_dim2_kernel[grid](x, out, B, M, N, BLOCK_M=128, BLOCK_K=64, num_warps=4, num_stages=2)
            return out
        if dim == 0:
            out = torch.empty((1, M, N), device=x.device, dtype=x.dtype)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
            _reduce_dim0_kernel[grid](x, out, B, M, N, BLOCK_M=64, BLOCK_N=128, BLOCK_KB=16, num_warps=4, num_stages=2)
            return out
        raise AssertionError("Unreachable reduction dimension.")
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='npu')
    return [x]
def get_init_inputs():
    return [reduce_dim]
