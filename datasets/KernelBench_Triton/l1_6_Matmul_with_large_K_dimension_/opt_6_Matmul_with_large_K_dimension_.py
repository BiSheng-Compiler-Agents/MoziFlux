import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.runtime.driver as driver
import triton.language.extra.cann.extension as al

_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 64
_SPLIT_THRESHOLD_K = 8192
_MAX_SPLIT_K = 32
_REDUCE_BLOCK = 512


@triton.jit
def _matmul_direct_kernel(
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
    offs_k_base = tl.arange(0, BLOCK_K)
    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)
    tl.multiple_of(offs_k_base, 16)
    tl.max_contiguous(offs_k_base, BLOCK_K)
    row_mask = offs_m[:, None] < M
    col_mask = offs_n[None, :] < N
    c_mask = row_mask & col_mask
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in tl.range(0, num_k_tiles):
        offs_k = k_tile * BLOCK_K + offs_k_base
        k_mask = offs_k < K
        a_ptrs = a_ptr + offs_m[:,
                                None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:,
                                None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs,
                    mask=row_mask & k_mask[None, :],
                    other=0.0,
                    care_padding=False)
        b = tl.load(b_ptrs,
                    mask=k_mask[:, None] & col_mask,
                    other=0.0,
                    care_padding=False)
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _matmul_splitk_partial_kernel(
    a_ptr,
    b_ptr,
    partial_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
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
    offs_k_base = tl.arange(0, BLOCK_K)
    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)
    tl.multiple_of(offs_k_base, 16)
    tl.max_contiguous(offs_k_base, BLOCK_K)
    row_mask = offs_m[:, None] < M
    col_mask = offs_n[None, :] < N
    c_mask = row_mask & col_mask
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    total_k_tiles = tl.cdiv(K, BLOCK_K)
    tiles_per_split = tl.cdiv(total_k_tiles, SPLIT_K)
    tile_begin = pid_k * tiles_per_split
    tile_end = tl.minimum(tile_begin + tiles_per_split, total_k_tiles)
    for k_tile in tl.range(tile_begin, tile_end):
        offs_k = k_tile * BLOCK_K + offs_k_base
        k_mask = offs_k < K
        a_ptrs = a_ptr + offs_m[:,
                                None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:,
                                None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs,
                    mask=row_mask & k_mask[None, :],
                    other=0.0,
                    care_padding=False)
        b = tl.load(b_ptrs,
                    mask=k_mask[:, None] & col_mask,
                    other=0.0,
                    care_padding=False)
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc)
    p_ptrs = partial_ptr + pid_k * M * N + offs_m[:,
                                                  None] * N + offs_n[None, :]
    tl.store(p_ptrs, acc, mask=c_mask)


@triton.jit
def _splitk_reduce_kernel(
    partial_ptr,
    c_ptr,
    total: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_e = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_s = tl.arange(0, BLOCK_S)
    mask = offs_e < total
    vals = tl.load(partial_ptr + offs_s[:, None] * total + offs_e[None, :],
                   mask=(offs_s[:, None] < SPLIT_K) & mask[None, :],
                   other=0.0)
    out = tl.sum(vals, axis=0)
    tl.store(c_ptr + offs_e, out, mask=mask)


def _num_aicores(device):
    try:
        dev = torch.npu.current_device()
        return int(
            driver.active.utils.get_device_properties(dev)["num_aicore"])
    except Exception:
        return 20


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


class ModelNew(nn.Module):
    """Matrix multiplication C = A @ B optimized for large K on Ascend NPU."""

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
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        grid_m = triton.cdiv(M, _BLOCK_M)
        grid_n = triton.cdiv(N, _BLOCK_N)

        if K >= _SPLIT_THRESHOLD_K:
            # Split-K exposes parallelism for the benchmark's huge K; partials avoid fp32 atomic_add instability.
            split_k = min(16, triton.cdiv(K, _BLOCK_K))
            partial = torch.empty((split_k, M, N),
                                  device=A.device,
                                  dtype=A.dtype)
            _matmul_splitk_partial_kernel[(grid_m, grid_n, split_k)](
                A,
                B,
                partial,
                M,
                N,
                K,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(1),
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                BLOCK_K=_BLOCK_K,
                SPLIT_K=split_k,
            )
            block_s = _next_power_of_2(split_k)
            _splitk_reduce_kernel[(triton.cdiv(M * N, _REDUCE_BLOCK), )](
                partial,
                C,
                M * N,
                M,
                N,
                SPLIT_K=split_k,
                BLOCK_E=_REDUCE_BLOCK,
                BLOCK_S=block_s,
            )
        else:
            _matmul_direct_kernel[(grid_m, grid_n)](
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
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                BLOCK_K=_BLOCK_K,
            )
        return C


M = 256
N = 256
K = 131072 * 4


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []
