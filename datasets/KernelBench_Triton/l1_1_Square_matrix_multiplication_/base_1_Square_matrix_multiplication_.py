import os

import torch
import torch.nn as nn

import triton
import triton.language as tl

EXACT_N = 4096
EXACT_BLOCK_M = 128
EXACT_BLOCK_N = 128
EXACT_BLOCK_K = 32
EXACT_GROUP_M = 4


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < m
    n_mask = offs_n < n

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_mask = (k_start + offs_k) < k
        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        tl.compile_hint(a, "dot_pad_only_k")
        tl.compile_hint(b, "dot_pad_only_k")
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _matmul_kernel_exact(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NUM_PID_M: tl.constexpr,
    NUM_PID_N: tl.constexpr,
    EXACT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    group_width = GROUP_M * NUM_PID_N
    group_id = pid // group_width
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(NUM_PID_M - first_pid_m, GROUP_M)
    pid_in_group = pid % group_width
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, EXACT_K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        tl.compile_hint(a, "dot_pad_only_k")
        tl.compile_hint(b, "dot_pad_only_k")
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)


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


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("ModelNew expects two 2D tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {tuple(a.shape)} and {tuple(b.shape)}.")
    if a.shape[0] != a.shape[1] or b.shape[0] != b.shape[1]:
        raise ValueError("This operator is defined for square matrix multiplication inputs.")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Square matrices must share the same inner dimension.")
    if a.device != b.device:
        raise ValueError("Inputs must be on the same device.")
    if a.dtype != b.dtype:
        raise ValueError("Inputs must have the same dtype.")
    if a.dtype not in {torch.float16, torch.float32}:
        raise TypeError(f"Unsupported dtype for Triton matmul: {a.dtype}.")
    _require_supported_runtime(a)
    return a.contiguous(), b.contiguous()


def _triton_square_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _validate_inputs(a, b)
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    if m == EXACT_N and n == EXACT_N and k == EXACT_N:
        num_pid_m = EXACT_N // EXACT_BLOCK_M
        num_pid_n = EXACT_N // EXACT_BLOCK_N
        grid = (num_pid_m * num_pid_n,)
        _matmul_kernel_exact[grid](
            a,
            b,
            c,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            NUM_PID_M=num_pid_m,
            NUM_PID_N=num_pid_n,
            EXACT_K=EXACT_N,
            BLOCK_M=EXACT_BLOCK_M,
            BLOCK_N=EXACT_BLOCK_N,
            BLOCK_K=EXACT_BLOCK_K,
            GROUP_M=EXACT_GROUP_M,
        )
        return c.to(dtype=a.dtype)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )
    return c.to(dtype=a.dtype)


class ModelNew(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return _triton_square_matmul(A, B)
N = 2048 * 2

def get_inputs():
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    A = torch.rand(N, N, device=device)
    B = torch.rand(N, N, device=device)
    return [A, B]
def get_init_inputs():
    return []  # No special initialization inputs needed
