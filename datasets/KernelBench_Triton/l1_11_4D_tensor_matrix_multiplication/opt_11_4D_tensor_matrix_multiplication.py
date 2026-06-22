import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Ascend CANN extension for compiler hints (dot_pad_only_k, multibuffer, etc.)
try:
    import triton.language.extra.cann.extension as al

    HAS_CANN_EXT = True
except ImportError:
    HAS_CANN_EXT = False


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "HAS_CANN_EXT": HAS_CANN_EXT
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "HAS_CANN_EXT": HAS_CANN_EXT
            },
            num_stages=4,
            num_warps=8),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "HAS_CANN_EXT": HAS_CANN_EXT
            },
            num_stages=4,
            num_warps=8),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 8,
                "HAS_CANN_EXT": HAS_CANN_EXT
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
                "HAS_CANN_EXT": HAS_CANN_EXT
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "HAS_CANN_EXT": HAS_CANN_EXT
            },
            num_stages=3,
            num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_2d_kernel(
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
    HAS_CANN_EXT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Persistent 1D-grid matmul kernel with GROUP_M swizzle.
    Each program iterates over multiple tiles via work-stealing when the
    total tile count exceeds the Ascend 65535-coreDim limit.
    """
    # Total tiles needed
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    # Persistent 1D grid with contiguous tile assignment per program
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = num_pid_m * num_pid_n
    tiles_per_program = tl.cdiv(num_tiles, num_programs)
    start_pid = pid * tiles_per_program
    end_pid = min(start_pid + tiles_per_program, num_tiles)

    for tile_pid in range(start_pid, end_pid):
        # GROUP_M swizzle: re-map 1D tile_pid to 2D (pid_m, pid_n)
        group_id = tile_pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_pid % group_size_m)
        pid_n = (tile_pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Boundary masks
        row_mask = offs_m[:, None] < M
        col_mask = offs_n[None, :] < N
        c_mask = row_mask & col_mask

        # FP32 accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K loop with tl.range
        num_k_iters = tl.cdiv(K, BLOCK_K)
        for k_idx in tl.range(0, num_k_iters):
            k_offs = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)

            a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                              k_offs[None, :] * stride_ak)
            b_ptrs = B_ptr + (k_offs[:, None] * stride_bk +
                              offs_n[None, :] * stride_bn)

            a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

            a = tl.load(a_ptrs, mask=a_mask, other=0.0, care_padding=False)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, care_padding=False)

            if HAS_CANN_EXT:
                al.compile_hint(a, "dot_pad_only_k")
                al.compile_hint(b, "dot_pad_only_k")

            acc = tl.dot(a, b, acc)

        # Store output
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                          offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc, mask=c_mask)

        # Steal next tile
        pid += num_programs


def _matmul_triton(A2d: torch.Tensor, B: torch.Tensor,
                   out_dtype: torch.dtype) -> torch.Tensor:
    """
    Computes C = A2d @ B using an optimized Triton kernel.
    A2d: (M, K)
    B:   (K, N)
    Returns C: (M, N) in dtype=out_dtype
    """
    assert A2d.dim() == 2 and B.dim() == 2
    M, K = A2d.shape
    Kb, N = B.shape
    assert K == Kb, "Inner dimensions must match for matmul"

    A_ptr = A2d
    B_ptr = B

    # Allocate output
    C = torch.empty((M, N), device=A2d.device, dtype=out_dtype)

    # Strides in units of elements
    stride_am, stride_ak = A2d.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Persistent 1D grid capped at 65535 for Ascend
    MAX_COREDIM = 65535
    num_pid_m = triton.cdiv(M, 128)  # approximate max with smallest BLOCK_M
    num_pid_n = triton.cdiv(N, 128)
    total_tiles = num_pid_m * num_pid_n
    num_programs = min(total_tiles, MAX_COREDIM)

    def grid(meta):
        return (num_programs, )

    _matmul_2d_kernel[grid](
        A_ptr,
        B_ptr,
        C,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
    )
    return C


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


def _validate_inputs(A: torch.Tensor,
                     B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if A.dim() != 4 or B.dim() != 2:
        raise ValueError("ModelNew expects a 4D tensor and a 2D matrix.")
    if A.shape[-1] != B.shape[0]:
        raise ValueError(
            f"Incompatible shapes for tensor-matrix multiplication: {tuple(A.shape)} and {tuple(B.shape)}."
        )
    if A.device != B.device:
        raise ValueError("Inputs must be on the same device.")
    if A.dtype != B.dtype:
        raise ValueError("Inputs must have the same dtype.")
    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype for Triton tensor-matrix multiplication: {A.dtype}."
        )
    _require_supported_runtime(A)
    return A.contiguous(), B.contiguous()


def _tensor_matrix_multiply(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A, B = _validate_inputs(A, B)
    b, i, j, length = A.shape
    _, k = B.shape
    A2d = A.reshape(-1, length)
    out_dtype = torch.result_type(A, B)
    C2d = _matmul_triton(A2d, B, out_dtype)
    return C2d.view(b, i, j, k)


class ModelNew(nn.Module):
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return _tensor_matrix_multiply(A, B)


b = 8
i = 256
j = 512
length = 256
k = 768


def get_inputs():
    device = "npu" if hasattr(torch,
                              "npu") and torch.npu.is_available() else "cpu"
    A = torch.rand(b, i, j, length, device=device)
    B = torch.rand(length, k, device=device)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
