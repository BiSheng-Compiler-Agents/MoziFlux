import os

import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    ],
    key=["N"],
)
@triton.jit
def _lower_tri_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    if m0 >= N or n0 >= N:
        return
    if n0 > (m0 + BLOCK_M - 1):
        return

    rm = m0 + tl.arange(0, BLOCK_M)
    rn = n0 + tl.arange(0, BLOCK_N)
    m_in = rm < N
    n_in = rn < N
    a_row_base = A_ptr + rm[:, None] * stride_am
    b_col_base = B_ptr + rn[None, :] * stride_bn
    a_row_legal = rm[:, None]
    b_col_legal = rn[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    rk = tl.arange(0, BLOCK_K)
    k_start = n0
    k_limit = tl.minimum(N, m0 + BLOCK_M)

    tl.multiple_of(rm, BLOCK_M)
    tl.multiple_of(rn, BLOCK_N)
    tl.multiple_of(rk, BLOCK_K)

    for k0 in range(k_start, k_limit, BLOCK_K):
        k = k0 + rk
        k_in = k < k_limit

        a_ptrs = a_row_base + k[None, :] * stride_ak
        b_ptrs = b_col_base + k[:, None] * stride_bk

        a_mask = m_in[:, None] & k_in[None, :] & (k[None, :] <= a_row_legal)
        b_mask = k_in[:, None] & n_in[None, :] & (b_col_legal <= k[:, None])

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tile_all_lower = (n0 + BLOCK_N - 1) <= m0
    full_in_bounds = (m0 + BLOCK_M) <= N and (n0 + BLOCK_N) <= N

    if tile_all_lower and full_in_bounds:
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty))
    else:
        store_mask = (rm[:, None] >= rn[None, :]) & m_in[:, None] & n_in[None, :]
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=store_mask)


def _require_supported_runtime(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        return
    if tensor.device.type == "npu":
        return
    if os.environ.get("TRITON_INTERPRET") == "1":
        return
    raise RuntimeError(
        "This operator requires CUDA or NPU tensors, or TRITON_INTERPRET=1 for Triton interpreter mode."
    )


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("ModelNew expects two 2D tensors.")
    if a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("ModelNew expects square matrices of the same shape.")
    if a.device != b.device:
        raise ValueError("Inputs must be on the same device.")
    if a.dtype != b.dtype:
        raise ValueError("Inputs must have the same dtype.")
    if a.dtype not in {torch.float16, torch.float32}:
        raise TypeError(f"Unsupported dtype for lower triangular matmul: {a.dtype}.")
    _require_supported_runtime(a)
    return a.contiguous(), b.contiguous()


class ModelNew(nn.Module):
    """
    Performs lower-triangular matrix multiplication for square inputs.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A, B = _validate_inputs(A, B)
        N = A.shape[0]
        C = torch.zeros((N, N), device=A.device, dtype=A.dtype)

        grid = lambda META: (
            triton.cdiv(N, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        _lower_tri_matmul_kernel[grid](
            A,
            B,
            C,
            N,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
        )
        return C
M = 4096

def get_inputs():
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    A = torch.rand(M, M, device=device)
    B = torch.rand(M, M, device=device)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]
def get_init_inputs():
    return []  # No special initialization inputs needed
