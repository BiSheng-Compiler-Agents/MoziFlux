import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Favor large N-tiles to improve coalescing on B (row-major, stride_n = 1)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        # Alternative shapes for different M/N divisibility and occupancy trade-offs
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=1),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_smallk_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this CTA
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    rk = tl.arange(0, BLOCK_K)                    # [BK]

    # Hints to help vectorization on contiguous axes
    tl.max_contiguous(rn, BLOCK_N)
    tl.max_contiguous(rk, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K
    for k0 in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + rm[:, None] * stride_am + (k0 + rk)[None, :] * stride_ak  # [BM, BK]
        b_ptrs = B_ptr + (k0 + rk)[:, None] * stride_bk + rn[None, :] * stride_bn  # [BK, BN]

        a_mask = (rm[:, None] < M) & ((k0 + rk)[None, :] < K)
        b_mask = ((k0 + rk)[:, None] < K) & (rn[None, :] < N)

        # Load tiles, OOB masked to 0
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # FP32 matmul (disable TF32 to match torch.matmul FP32 semantics closely)
        acc += tl.dot(a, b, allow_tf32=False)

    # Write back
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
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

        # Ensure contiguity for predictable strides
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Output
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Grid: one program per output tile
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

        _matmul_smallk_kernel[grid](
            A_c, B_c, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            B_c.stride(0), B_c.stride(1),
            C.stride(0), C.stride(1),
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
    return []  # No special initialization inputs needed