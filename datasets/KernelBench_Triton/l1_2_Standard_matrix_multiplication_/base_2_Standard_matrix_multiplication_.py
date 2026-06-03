import torch
import torch.nn as nn
import triton
import triton.language as tl

import torch_npu  # noqa: F401


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
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
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_width = GROUP_M * num_pid_n
    group_id = pid // group_width
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % group_width) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k_start + offs_k)[None, :] < K)
        b_mask = ((k_start + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2D tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {tuple(a.shape)} @ {tuple(b.shape)}.")
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("ModelNew expects Ascend NPU tensors.")
    if a.device != b.device:
        raise ValueError("A and B must be on the same device.")
    if a.dtype != b.dtype:
        raise ValueError("A and B must have the same dtype.")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype for Triton matmul: {a.dtype}.")
    if a.requires_grad or b.requires_grad:
        raise RuntimeError("ModelNew does not support autograd-tracked inputs.")
    return a.contiguous(), b.contiguous()


def _matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _validate_inputs(a, b)
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
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
    )
    return c if a.dtype == torch.float32 else c.to(dtype=a.dtype)


class ModelNew(nn.Module):
    """
    Triton implementation of a standard 2D matrix multiplication C = A @ B.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _matmul_triton(A, B)
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    A = torch.rand(M, K, device=device)
    B = torch.rand(K, N, device=device)
    return [A, B]
def get_init_inputs():
    return []  # No special initialization inputs needed
