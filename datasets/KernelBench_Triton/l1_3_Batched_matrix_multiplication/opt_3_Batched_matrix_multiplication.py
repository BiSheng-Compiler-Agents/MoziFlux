import torch
import torch.nn as nn

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.autotune(
    configs=[
        # Wide tiles — best for large M, N
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 8
        }),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
            "GROUP_M": 8
        }),
        # Tall M, narrow N
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_M": 8
        }),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 8
        }),
        # Wide M, tall N
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 8
        }),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32,
            "GROUP_M": 8
        }),
        # Square-ish, balanced
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 128,
            "GROUP_M": 8
        }),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_M": 4
        }),
        # Large unbalanced — for very wide or very tall matrices
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 4
        }),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 256,
            "BLOCK_K": 32,
            "GROUP_M": 4
        }),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BATCH,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    # Compute block indices with GROUP_M swizzle for L2 reuse
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer alignment hints for better codegen
    a_ptr = tl.multiple_of(a_ptr, BLOCK_K)
    b_ptr = tl.multiple_of(b_ptr, BLOCK_N)

    # Base pointers for the current batch
    a_ptr_batch = a_ptr + bid * stride_ab
    b_ptr_batch = b_ptr + bid * stride_bb
    c_ptr_batch = c_ptr + bid * stride_cb

    # Hoisted masks — computed once outside the K loop
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    # Accumulator in FP32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Use tl.range for better compiler loop pipelining
    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k_idx in tl.range(0, num_k_iters):
        k_offs = k_idx * BLOCK_K + offs_k

        # Separate k_mask orientation for A (k as columns) vs B (k as rows)
        k_mask_a = k_offs[None, :] < K
        k_mask_b = k_offs[:, None] < K

        a_ptrs = a_ptr_batch + (offs_m[:, None] * stride_am +
                                k_offs[None, :] * stride_ak)
        b_ptrs = b_ptr_batch + (k_offs[:, None] * stride_bk +
                                offs_n[None, :] * stride_bn)

        a_mask = m_mask & k_mask_a
        b_mask = k_mask_b & n_mask

        a = tl.load(a_ptrs, mask=a_mask, other=0.0, care_padding=False)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, care_padding=False)

        # Ascend-specific compiler hint — reduces UB padding overhead
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")

        # In-place accumulation — saves one tile-size temp buffer vs acc += tl.dot(a, b)
        acc = tl.dot(a, b, acc)

    c_ptrs = c_ptr_batch + (offs_m[:, None] * stride_cm +
                            offs_n[None, :] * stride_cn)
    c_mask = m_mask & n_mask
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C
    have the same batch dimension. Supports 2D and 3D inputs with broadcasting.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k) or (m, k).
            B: Input tensor of shape (batch_size, k, n) or (k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n) or (m, n).
        """
        # Handle 2D inputs by adding a batch dimension
        if A.dim() == 2 and B.dim() == 2:
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)
        elif A.dim() == 2 and B.dim() == 3:
            A = A.unsqueeze(0).expand(B.shape[0], -1, -1).contiguous()
        elif A.dim() == 3 and B.dim() == 2:
            B = B.unsqueeze(0).expand(A.shape[0], -1, -1).contiguous()
        elif A.dim() != 3 or B.dim() != 3:
            raise ValueError("A and B must be 2D or 3D tensors")

        if A.dtype != B.dtype:
            raise TypeError("A and B must have the same dtype")
        if A.device.type != "npu" or B.device.type != "npu":
            raise ValueError("ModelNew expects Ascend NPU tensors")
        if A.device != B.device:
            raise ValueError("A and B must be on the same device")
        if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "ModelNew supports float16, bfloat16, and float32 inputs")

        BATCH, M, K = A.shape
        BATCH_B, K_B, N = B.shape
        if BATCH != BATCH_B or K != K_B:
            raise ValueError(
                "A and B must satisfy A.shape == (batch, m, k) and B.shape == (batch, k, n)"
            )

        # Make contiguous for predictable strides/coalescing
        A_ = A.contiguous()
        B_ = B.contiguous()

        # Allocate output
        C = torch.empty((BATCH, M, N), device=A.device, dtype=A.dtype)

        # Strides in elements
        stride_ab, stride_am, stride_ak = A_.stride()
        stride_bb, stride_bk, stride_bn = B_.stride()
        stride_cb, stride_cm, stride_cn = C.stride()

        # Grid: all tiles across (M, N) for each batch
        def grid(META):
            return (
                triton.cdiv(M, META["BLOCK_M"]) *
                triton.cdiv(N, META["BLOCK_N"]),
                BATCH,
            )

        _bmm_kernel[grid](
            A_,
            B_,
            C,
            BATCH,
            M,
            N,
            K,
            stride_ab,
            stride_am,
            stride_ak,
            stride_bb,
            stride_bk,
            stride_bn,
            stride_cb,
            stride_cm,
            stride_cn,
        )
        return C


batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4


def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
