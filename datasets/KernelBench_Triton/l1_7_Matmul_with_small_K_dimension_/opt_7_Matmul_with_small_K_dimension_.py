import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

try:
    import triton.language.extra.cann.extension as al
    HAS_AL = True
except (ImportError, ModuleNotFoundError):
    HAS_AL = False


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=2),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_smallk_opt_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk_base = tl.arange(0, BLOCK_K)

    tl.max_contiguous(rn, BLOCK_N)
    tl.max_contiguous(rk_base, BLOCK_K)

    row_mask = rm[:, None] < M
    col_mask = rn[None, :] < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k_idx in tl.range(0, num_k_iters):
        rk = rk_base + k_idx * BLOCK_K
        k_mask = rk < K
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        a = tl.load(a_ptrs,
                    mask=row_mask & k_mask[None, :],
                    other=0.0,
                    care_padding=False)
        b = tl.load(b_ptrs,
                    mask=k_mask[:, None] & col_mask,
                    other=0.0,
                    care_padding=False)
        if HAS_AL:
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc)

    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=row_mask & col_mask)


class ModelNew(nn.Module):
    """C = A @ B for 2D float32 Ascend NPU tensors, optimized for small K."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("ModelNew expects two 2D tensors.")
        M, K = A.shape
        Kb, N = B.shape
        if K != Kb:
            raise ValueError("Inner dimensions must match.")
        if A.device != B.device:
            raise ValueError("Inputs must be on the same device.")
        if A.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU.")
        if A.dtype != B.dtype:
            raise TypeError("Inputs must have the same dtype.")
        if A.dtype != torch.float32:
            raise TypeError("ModelNew only supports torch.float32 inputs.")

        A_c = A.contiguous()
        B_c = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        def grid(META):
            return (
                triton.cdiv(M, META["BLOCK_M"]),
                triton.cdiv(N, META["BLOCK_N"]),
            )

        _matmul_smallk_opt_kernel[grid](
            A_c,
            B_c,
            C,
            M,
            N,
            K,
            A_c.stride(0),
            A_c.stride(1),
            B_c.stride(0),
            B_c.stride(1),
            C.stride(0),
            C.stride(1),
        )
        return C


M = 16384 * 2
N = 16384 * 2
K = 32 * 2


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []
