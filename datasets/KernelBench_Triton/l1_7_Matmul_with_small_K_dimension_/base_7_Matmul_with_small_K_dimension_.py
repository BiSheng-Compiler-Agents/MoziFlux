# Round 24: Copy of round-17 + tl.max_contiguous(rk, BLOCK_K) and tl.max_contiguous(tl.arange(0, BLOCK_N), BLOCK_N)
import torch
import torch.nn as nn
import triton
import triton.language as tl

_BLOCK_M = 128
_BLOCK_N = 256
_BLOCK_K = 64


@triton.jit
def _matmul_smallk_kernel(
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
    BLOCK_M: tl.constexpr = _BLOCK_M,
    BLOCK_N: tl.constexpr = _BLOCK_N,
    BLOCK_K: tl.constexpr = _BLOCK_K,
):
    pid_m = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)

    tl.max_contiguous(rk, BLOCK_K)

    k0 = 0
    a_ptrs = A_ptr + rm[:, None] * stride_am + (k0 + rk)[None, :] * stride_ak
    a_mask = (rm[:, None] < M) & ((k0 + rk)[None, :] < K)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

    for n_start in range(0, N, BLOCK_N * 2):
        tl.max_contiguous(tl.arange(0, BLOCK_N), BLOCK_N)

        rn0 = n_start + tl.arange(0, BLOCK_N)
        b0_ptrs = B_ptr + rn0[:, None] * stride_bk + (k0 +
                                                      rk)[None, :] * stride_bn
        b0_mask = (rn0[:, None] < N) & ((k0 + rk)[None, :] < K)
        b0 = tl.load(b0_ptrs, mask=b0_mask, other=0.0).to(tl.float32)

        rn1 = rn0 + BLOCK_N
        b1_ptrs = B_ptr + rn1[:, None] * stride_bk + (k0 +
                                                      rk)[None, :] * stride_bn
        b1_mask = (rn1[:, None] < N) & ((k0 + rk)[None, :] < K)
        b1 = tl.load(b1_ptrs, mask=b1_mask, other=0.0).to(tl.float32)

        acc0 = tl.dot(a, tl.trans(b0), allow_tf32=False)
        c0_ptrs = C_ptr + rm[:, None] * stride_cm + rn0[None, :] * stride_cn
        c0_mask = (rm[:, None] < M) & (rn0[None, :] < N)
        tl.store(c0_ptrs, acc0, mask=c0_mask)

        acc1 = tl.dot(a, tl.trans(b1), allow_tf32=False)
        c1_ptrs = C_ptr + rm[:, None] * stride_cm + rn1[None, :] * stride_cn
        c1_mask = (rm[:, None] < M) & (rn1[None, :] < N)
        tl.store(c1_ptrs, acc1, mask=c1_mask)


class ModelNew(nn.Module):

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
        B_t = B.T.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        grid = (triton.cdiv(M, _BLOCK_M), )

        _matmul_smallk_kernel[grid](
            A_c,
            B_t,
            C,
            M,
            N,
            K,
            A_c.stride(0),
            A_c.stride(1),
            B_t.stride(0),
            B_t.stride(1),
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
