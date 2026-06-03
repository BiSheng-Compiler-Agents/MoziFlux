import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _symmetric_matmul_kernel(
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        a_mask = (offs_m[:, None] < m) & ((k_start + offs_k)[None, :] < k)
        b_mask = ((k_start + offs_k)[:, None] < k) & (offs_n[None, :] < n)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


def _require_supported_runtime(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        return
    if tensor.device.type == "npu":
        return
    if os.environ.get("TRITON_INTERPRET") == "1":
        return
    raise RuntimeError(
        "This operator requires CUDA or NPU tensors, or TRITON_INTERPRET=1."
    )


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("ModelNew expects two 2D tensors.")
    if a.shape[0] != a.shape[1] or b.shape[0] != b.shape[1]:
        raise ValueError("This operator expects square symmetric-matrix inputs.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {tuple(a.shape)} and {tuple(b.shape)}.")
    if a.device != b.device:
        raise ValueError("Inputs must be on the same device.")
    if a.dtype != b.dtype:
        raise ValueError("Inputs must have the same dtype.")
    if a.dtype not in {torch.float16, torch.float32}:
        raise TypeError(f"Unsupported dtype for Triton matmul: {a.dtype}.")
    if not torch.allclose(a, a.transpose(-1, -2)):
        raise ValueError("Input A must be symmetric.")
    if not torch.allclose(b, b.transpose(-1, -2)):
        raise ValueError("Input B must be symmetric.")
    _require_supported_runtime(a)
    return a.contiguous(), b.contiguous()


def _triton_symmetric_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _validate_inputs(a, b)
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    _symmetric_matmul_kernel[grid](
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
    Matrix multiplication for two symmetric square matrices.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _triton_symmetric_matmul(A, B)
N = 4096

def get_inputs():
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    A = torch.rand(N, N, device=device)
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.rand(N, N, device=device)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]
def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []