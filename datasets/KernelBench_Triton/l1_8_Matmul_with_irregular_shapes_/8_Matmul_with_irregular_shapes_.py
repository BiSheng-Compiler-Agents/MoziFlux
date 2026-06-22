import torch
import torch.nn as nn
import triton
import triton.language as tl


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
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_start in range(0, k, block_k):
        k_offsets = k_start + offs_k
        a_mask = (offs_m[:, None] < m) & (k_offsets[None, :] < k)
        b_mask = (k_offsets[:, None] < k) & (offs_n[None, :] < n)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


class ModelNew(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError("ModelNew expects two 2D tensors")
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Incompatible matmul shapes: {tuple(a.shape)} and {tuple(b.shape)}"
            )
        if a.device.type != "npu" or b.device.type != "npu":
            raise RuntimeError("ModelNew requires NPU tensors")
        if a.dtype != b.dtype:
            raise TypeError("ModelNew requires matching input dtypes")
        if a.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                "ModelNew supports float16 and bfloat16 inputs only")

        a_contig = a.contiguous()
        b_contig = b.contiguous()
        m, k = a_contig.shape
        _, n = b_contig.shape
        c = torch.empty((m, n), device=a_contig.device, dtype=a_contig.dtype)

        block_m = 128
        block_n = 128
        block_k = 32
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))

        _matmul_kernel[grid](
            a_contig,
            b_contig,
            c,
            m,
            n,
            k,
            a_contig.stride(0),
            a_contig.stride(1),
            b_contig.stride(0),
            b_contig.stride(1),
            c.stride(0),
            c.stride(1),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=8,
            num_stages=4,
        )
        return c


M = 8205
K = 2949
N = 5921


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
