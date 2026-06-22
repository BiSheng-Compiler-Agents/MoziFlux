import os

import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_warps=8,
                      num_stages=4),
    ],
    key=["N"],
)
@triton.jit
def _upper_tri_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    stride_Am,
    stride_Ak,
    stride_Bk,
    stride_Bn,
    stride_Cm,
    stride_Cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D tile ids
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # Out-of-range tiles can be skipped early
    if (m0 >= N) or (n0 >= N):
        return
    # If the whole tile is strictly below the diagonal, we can skip it.
    if m0 > (n0 + BLOCK_N - 1):
        return

    rm = m0 + tl.arange(0, BLOCK_M)
    rn = n0 + tl.arange(0, BLOCK_N)
    m_in = rm < N
    n_in = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    rk = tl.arange(0, BLOCK_K)
    tl.multiple_of(rm, BLOCK_M)
    tl.multiple_of(rn, BLOCK_N)
    tl.multiple_of(rk, BLOCK_K)

    a_base = A_ptr + (rm[:, None] * stride_Am)
    b_base = B_ptr + (rn[None, :] * stride_Bn)
    a_mask_m = m_in[:, None]
    b_mask_n = n_in[None, :]

    # K sweep using tiled dot-product.
    # Keep inputs in their native dtype to leverage tensor cores for fp16/bf16,
    # while accumulating in fp32. Disable TF32 to match PyTorch fp32 numerics.
    for k0 in range(0, N, BLOCK_K):
        k = k0 + rk
        k_in = k < N

        a_ptrs = a_base + (k[None, :] * stride_Ak)
        b_ptrs = b_base + (k[:, None] * stride_Bk)

        a = tl.load(a_ptrs, mask=a_mask_m & k_in[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_in[:, None] & b_mask_n, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = C_ptr + (rm[:, None] * stride_Cm + rn[None, :] * stride_Cn)

    # Store only upper-triangular region
    tile_all_upper = (m0 + BLOCK_M - 1) <= n0
    full_in_bounds = (m0 + BLOCK_M) <= N and (n0 + BLOCK_N) <= N

    if tile_all_upper and full_in_bounds:
        # Fast path: no masks needed
        tl.store(c_ptrs, acc)
    else:
        store_mask = (rm[:, None] <= rn[None, :]) & m_in[:,
                                                         None] & n_in[None, :]
        tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) and returns its upper triangular part.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs matrix multiplication and returns torch.triu(A @ B).

        Args:
            A (torch.Tensor): Matrix of shape (N, N).
            B (torch.Tensor): Matrix of shape (N, N).

        Returns:
            torch.Tensor: Upper triangular part of A @ B, shape (N, N).
        """
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("ModelNew expects two 2D tensors.")
        if A.shape != B.shape or A.shape[0] != A.shape[1]:
            raise ValueError(
                "ModelNew expects square matrices of the same shape.")
        if A.device != B.device:
            raise ValueError("Inputs must be on the same device.")
        if A.dtype != B.dtype:
            raise ValueError("Inputs must have the same dtype.")
        if A.dtype not in {torch.float16, torch.float32}:
            raise TypeError(
                f"Unsupported dtype for upper triangular matmul: {A.dtype}.")
        if not (A.is_cuda or A.device.type == "npu"
                or os.environ.get("TRITON_INTERPRET") == "1"):
            raise RuntimeError(
                "This operator requires CUDA or NPU tensors, or TRITON_INTERPRET=1 for Triton interpreter mode."
            )

        N = A.shape[0]
        A_ = A.contiguous()
        B_ = B.contiguous()

        # Allocate output as zeros so we can skip strictly-below-diagonal tiles safely
        C = torch.zeros((N, N), device=A.device, dtype=A.dtype)

        def grid(META):
            return (
                triton.cdiv(N, META["BLOCK_M"]),
                triton.cdiv(N, META["BLOCK_N"]),
            )

        _upper_tri_matmul_kernel[grid](
            A_,
            B_,
            C,
            N,
            A_.stride(0),
            A_.stride(1),
            B_.stride(0),
            B_.stride(1),
            C.stride(0),
            C.stride(1),
        )
        # C already contains only the upper-triangular values
        return C


N = 4096


def get_inputs():
    device = "npu" if hasattr(torch,
                              "npu") and torch.npu.is_available() else "cpu"
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.rand(N, N, device=device))
    B = torch.triu(torch.rand(N, N, device=device))
    return [A, B]


def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []
