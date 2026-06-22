import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

TARGET_K = 256
TARGET_N = 768
TARGET_BLOCK_M = 128
TARGET_BLOCK_N = 256
TARGET_BLOCK_K = 32
TARGET_GROUP_M = 16
TARGET_USE_HINTS = False


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 256,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_generic_kernel(
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
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_base = A_ptr + offs_m[:, None] * stride_am
    b_base = B_ptr + offs_n[None, :] * stride_bn
    a_mask_m = offs_m < M
    b_mask_n = offs_n < N

    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_base + k_range[None, :] * stride_ak
        b_ptrs = b_base + k_range[:, None] * stride_bk
        k_mask_row = k_range[None, :] < K
        k_mask_col = k_range[:, None] < K
        # Keep fp16/bf16 operands native for Ascend tl.dot; accumulator is fp32.
        a = tl.load(a_ptrs, mask=a_mask_m[:, None] & k_mask_row, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_col & b_mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)

    tl.store(c_ptrs, acc, mask=a_mask_m[:, None] & b_mask_n[None, :])


@triton.jit
def _matmul_target_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_HINTS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = 768 // BLOCK_N

    if GROUP_M > 1:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = A_ptr + offs_m[:, None] * stride_am
    b_base = B_ptr + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.static_range(0, 256, BLOCK_K):
        # Keep fp16/bf16 operands native for Ascend tl.dot; accumulator is fp32.
        a = tl.load(a_base + (k0 + offs_k)[None, :] * stride_ak)
        b = tl.load(b_base + (k0 + offs_k)[:, None] * stride_bk)
        if USE_HINTS:
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(b, "dot_pad_only_k")
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)


def _launch_target_kernel(A2d: torch.Tensor, B: torch.Tensor,
                          out_dtype: torch.dtype) -> torch.Tensor:
    M, _K = A2d.shape
    C = torch.empty((M, TARGET_N), device=A2d.device, dtype=out_dtype)
    stride_am, stride_ak = A2d.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    grid = (triton.cdiv(M, TARGET_BLOCK_M) * (TARGET_N // TARGET_BLOCK_N), )

    _matmul_target_kernel[grid](
        A2d,
        B,
        C,
        M,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=TARGET_BLOCK_M,
        BLOCK_N=TARGET_BLOCK_N,
        BLOCK_K=TARGET_BLOCK_K,
        GROUP_M=TARGET_GROUP_M,
        USE_HINTS=TARGET_USE_HINTS,
    )
    return C


def _matmul_triton(A2d: torch.Tensor, B: torch.Tensor,
                   out_dtype: torch.dtype) -> torch.Tensor:
    assert A2d.dim() == 2 and B.dim() == 2
    M, K = A2d.shape
    Kb, N = B.shape
    assert K == Kb, "Inner dimensions must match for matmul"

    if (K == TARGET_K and N == TARGET_N and M % TARGET_BLOCK_M == 0
            and TARGET_N % TARGET_BLOCK_N == 0
            and TARGET_K % TARGET_BLOCK_K == 0):
        return _launch_target_kernel(A2d, B, out_dtype)

    C = torch.empty((M, N), device=A2d.device, dtype=out_dtype)
    stride_am, stride_ak = A2d.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    _matmul_generic_kernel[grid](
        A2d,
        B,
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
    return []
