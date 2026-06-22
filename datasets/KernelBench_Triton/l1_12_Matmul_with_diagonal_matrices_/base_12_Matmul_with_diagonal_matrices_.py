import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _row_scale_kernel(
    a_ptr,  # *A: shape [N]
    b_ptr,  # *B: shape [N, M]
    c_ptr,  # *C: shape [N, M]
    N,
    M,
    stride_bm,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row block id
    pid_n = tl.program_id(1)  # block id along columns

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_start = pid_n * BLOCK_N
    offs_n = col_start + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    # Vectorization/alignment hints for better coalescing
    tl.max_contiguous(offs_n, BLOCK_N)
    tl.multiple_of(offs_n, 8)
    tl.static_assert(BLOCK_N % 8 == 0)
    tl.max_contiguous(rows, BLOCK_M)

    # Pointers for B and C tiles
    b_ptrs = b_ptr + rows[:, None] * stride_bm + offs_n[None, :] * stride_bn
    c_ptrs = c_ptr + rows[:, None] * stride_cm + offs_n[None, :] * stride_cn
    # Detect full interior tile along the column dimension to avoid predication.
    full_n = (col_start + BLOCK_N) <= M
    full_m = (pid_m * BLOCK_M + BLOCK_M) <= N
    full_tile = full_m & full_n

    if full_tile:
        a_vals = tl.load(a_ptr + rows, cache_modifier=".ca")
        b = tl.load(b_ptrs, cache_modifier=".cg")
        tl.store(c_ptrs, b * a_vals[:, None])
    else:
        cols_mask = offs_n < M
        mask = row_mask[:, None] & cols_mask[None, :]
        a_vals = tl.load(a_ptr + rows,
                         mask=row_mask,
                         other=0,
                         cache_modifier=".ca")
        b = tl.load(b_ptrs, mask=mask, other=0, cache_modifier=".cg")
        tl.store(c_ptrs, b * a_vals[:, None], mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        if A.dim() != 1 or B.dim() != 2:
            raise ValueError("Expected A to be 1D and B to be 2D.")
        if A.shape[0] != B.shape[0]:
            raise ValueError("A and B must have the same leading dimension.")
        if A.device.type != "npu" or B.device.type != "npu":
            raise ValueError("ModelNew expects Ascend NPU tensors.")

        N = A.shape[0]
        M = B.shape[1]

        # Match PyTorch matmul dtype promotion rules
        out_dtype = torch.result_type(A, B)
        A_cast = A.contiguous().to(out_dtype)
        B_cast = B.contiguous().to(out_dtype)
        C = torch.empty((N, M), device=B_cast.device, dtype=out_dtype)

        # Batch a few rows per program to amortize row-level launch overhead.
        BLOCK_M = 16
        BLOCK_N = 1024

        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_M']),
                    triton.cdiv(M, meta['BLOCK_N']))

        _row_scale_kernel[grid](
            A_cast,
            B_cast,
            C,
            N,
            M,
            B_cast.stride(0),
            B_cast.stride(1),
            C.stride(0),
            C.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=2,
        )
        return C


M = 4096
N = 4096


def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
