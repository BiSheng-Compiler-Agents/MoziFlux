import torch
import torch.nn as nn
import triton
import triton.language as tl

import torch_npu  # noqa: F401


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_AT_BT_kernel(
    A_ptr,  # A: (K, M)
    B_ptr,  # B: (N, K)
    C_ptr,  # C: (M, N) = A.T @ B.T
    M,
    N,
    K,
    stride_a_k,
    stride_a_m,
    stride_b_n,
    stride_b_k,
    stride_c_m,
    stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(offs_k, BLOCK_K)

    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)
    tl.multiple_of(offs_k, 16)
    tl.static_assert(BLOCK_K % 16 == 0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = A_ptr + (offs_k[:, None] * stride_a_k + offs_m[None, :] * stride_a_m)
    b_ptrs = B_ptr + (offs_n[None, :] * stride_b_n + offs_k[:, None] * stride_b_k)

    m_mask = offs_m < M
    n_mask = offs_n < N

    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a_km = tl.load(
            a_ptrs,
            mask=k_mask[:, None] & m_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        b_kn = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        acc += tl.dot(tl.trans(a_km), b_kn, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_a_k
        b_ptrs += BLOCK_K * stride_b_k

    c_ptrs = C_ptr + (offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n)
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _matmul_at_bt_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    K, M = A.shape
    N, K_b = B.shape
    if K != K_b:
        raise ValueError("Inner dimensions must match: A.shape[0] == B.shape[1]")

    A_ = A.contiguous()
    B_kn = B.transpose(0, 1).contiguous()
    C = torch.empty((M, N), device=A_.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _matmul_AT_BT_kernel[grid](
        A_,
        B_kn,
        C,
        M,
        N,
        K,
        A_.stride(0),
        A_.stride(1),
        B_kn.stride(1),
        B_kn.stride(0),
        C.stride(0),
        C.stride(1),
    )
    return C


class ModelNew(nn.Module):
    """
    Triton implementation of C = A.T @ B.T for A shaped (K, M) and B shaped (N, K).
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must both be 2D tensors")
        if A.shape[0] != B.shape[1]:
            raise ValueError("A.shape[0] must equal B.shape[1] for A.T @ B.T")
        if A.dtype != B.dtype:
            raise ValueError("A and B must have the same dtype")
        if A.device.type != "npu" or B.device.type != "npu":
            raise ValueError("A and B must be placed on Ascend NPU")
        if A.device != B.device:
            raise ValueError("A and B must be on the same device")
        if A.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Only float16 and bfloat16 inputs are supported")
        return _matmul_at_bt_triton(A, B)
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(K, M)
    B = torch.rand(N, K)
    return [A, B]
def get_init_inputs():
    return []  # No special initialization inputs needed
