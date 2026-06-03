import torch
import torch.nn as nn
import triton
import triton.language as tl


TARGET_M = 256
TARGET_N = 256
TARGET_K = 131072 * 4
TARGET_BLOCK_M = 64
TARGET_BLOCK_N = 128
TARGET_BLOCK_K = 256
TARGET_SPLIT_K = 8
TARGET_NUM_STAGES = 4
TARGET_NUM_WARPS = 8


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=5, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 128}, num_stages=5, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=6, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=6, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=6, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fallback_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    tl.multiple_of(offs_m, 8)
    tl.multiple_of(offs_n, 8)
    tl.multiple_of(offs_k, 8)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_start * BLOCK_K
        k_mask = (k_offset + offs_k) < K
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k_offset + offs_k)[None, :] * stride_ak)
        b_ptrs = b_ptr + ((k_offset + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=mask_m & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n, other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m & mask_n)


@triton.jit
def _target_shape_splitk_kernel(
    a_ptr,
    b_ptr,
    workspace_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ws,
    stride_wm,
    stride_wn,
    NUM_K_BLOCKS_PER_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)
    offs_k = tl.max_contiguous(offs_k, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (pid_k * BLOCK_K + offs_k)[None, :] * stride_ak
    b_ptrs = b_ptr + (pid_k * BLOCK_K + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_stride = BLOCK_K * SPLIT_K
    for _ in range(NUM_K_BLOCKS_PER_SPLIT):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += k_stride * stride_ak
        b_ptrs += k_stride * stride_bk
    workspace_ptrs = workspace_ptr + pid_k * stride_ws + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn
    tl.store(workspace_ptrs, acc)


@triton.jit
def _target_shape_reduce_kernel(
    workspace_ptr,
    c_ptr,
    stride_ws,
    stride_wm,
    stride_wn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_idx in range(SPLIT_K):
        partial_ptrs = workspace_ptr + split_idx * stride_ws + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn
        acc += tl.load(partial_ptrs)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("ModelNew expects 2D input tensors")
        if A.shape[1] != B.shape[0]:
            raise ValueError("Inner dimensions must match for matmul")
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            raise TypeError("ModelNew only supports float32 inputs")
        if A.device != B.device:
            raise ValueError("Input tensors must be on the same device")
        if A.device.type != "npu":
            raise ValueError("ModelNew requires Ascend NPU tensors")

        M, K = A.shape
        _, N = B.shape
        target_path = (
            M == TARGET_M
            and N == TARGET_N
            and K == TARGET_K
            and A.is_contiguous()
            and B.is_contiguous()
        )
        if target_path:
            workspace = torch.empty((TARGET_SPLIT_K, M, N), device=A.device, dtype=A.dtype)
            C = torch.empty((M, N), device=A.device, dtype=A.dtype)
            grid = (triton.cdiv(M, TARGET_BLOCK_M), triton.cdiv(N, TARGET_BLOCK_N), TARGET_SPLIT_K)
            num_k_blocks_per_split = triton.cdiv(K, TARGET_BLOCK_K * TARGET_SPLIT_K)
            _target_shape_splitk_kernel[grid](
                A,
                B,
                workspace,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(1),
                workspace.stride(0),
                workspace.stride(1),
                workspace.stride(2),
                NUM_K_BLOCKS_PER_SPLIT=num_k_blocks_per_split,
                BLOCK_M=TARGET_BLOCK_M,
                BLOCK_N=TARGET_BLOCK_N,
                BLOCK_K=TARGET_BLOCK_K,
                SPLIT_K=TARGET_SPLIT_K,
                num_warps=TARGET_NUM_WARPS,
                num_stages=TARGET_NUM_STAGES,
            )
            reduce_grid = (triton.cdiv(M, TARGET_BLOCK_M), triton.cdiv(N, TARGET_BLOCK_N))
            _target_shape_reduce_kernel[reduce_grid](
                workspace,
                C,
                workspace.stride(0),
                workspace.stride(1),
                workspace.stride(2),
                C.stride(0),
                C.stride(1),
                BLOCK_M=TARGET_BLOCK_M,
                BLOCK_N=TARGET_BLOCK_N,
                SPLIT_K=TARGET_SPLIT_K,
            )
            return C
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        _fallback_matmul_kernel[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
        )
        return C


M = TARGET_M
N = TARGET_N
K = TARGET_K


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []
