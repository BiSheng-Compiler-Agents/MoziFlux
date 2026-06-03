import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemv_rowblock_kernel(
    A_ptr,  # *A: (M, K)
    B_ptr,  # *B: (K, 1)
    C_ptr,  # *C: (M, 1)
    M: tl.constexpr,
    K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    row_offsets = rows[:, None] * stride_am

    k0 = 0
    while k0 < K:
        cols = k0 + k_offsets
        mask_k = cols < K
        a_ptrs = A_ptr + row_offsets + cols[None, :] * stride_ak
        b_ptrs = B_ptr + cols * stride_bk
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(a_tile.to(tl.float32) * b_tile.to(tl.float32)[None, :], axis=1)
        k0 += BLOCK_K

    c_ptrs = C_ptr + rows * stride_cm
    tl.store(c_ptrs, acc, mask=mask_m)


class ModelNew(nn.Module):
    """
    Matrix-vector multiplication (C = A * B) implemented with a Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("ModelNew expects 2D inputs A and B.")
        if B.shape[1] != 1:
            raise ValueError("ModelNew expects B to have shape (K, 1).")
        if A.shape[1] != B.shape[0]:
            raise ValueError("ModelNew requires A.shape[1] == B.shape[0].")
        if A.device.type != "npu" or B.device.type != "npu":
            raise ValueError("ModelNew requires both inputs to be on Ascend NPU.")
        if A.dtype != B.dtype:
            raise ValueError("ModelNew requires A and B to have the same dtype.")
        if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("ModelNew supports float16, bfloat16, and float32 inputs.")

        M, K = A.shape
        A_ctg = A.contiguous()
        B_ctg = B.contiguous()
        C = torch.empty((M, 1), device=A.device, dtype=A.dtype)

        BLOCK_M = 64
        BLOCK_K = 256

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        _gemv_rowblock_kernel[grid](
            A_ctg, B_ctg, C,
            M, K,
            A_ctg.stride(0), A_ctg.stride(1),
            B_ctg.stride(0), B_ctg.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )
        return C
M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]
def get_init_inputs():
    return []  # No special initialization inputs needed