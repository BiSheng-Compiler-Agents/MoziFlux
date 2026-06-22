import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


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
    group_m: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = tl.minimum(num_pid_m - first_pid_m, group_m)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    tl.multiple_of(offs_m, block_m)
    tl.multiple_of(offs_n, block_n)
    tl.static_assert(block_k % 16 == 0)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * block_m, 0),
        block_shape=(block_m, block_k),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * block_n),
        block_shape=(block_k, block_n),
        order=(1, 0),
    )

    for _ in range(0, k, block_k):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, block_k))
        b_block_ptr = tl.advance(b_block_ptr, (block_k, 0))

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


class ModelNew(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _launch(
        a_contig: torch.Tensor,
        b_contig: torch.Tensor,
        c: torch.Tensor,
        m: int,
        n: int,
        k: int,
        *,
        block_m: int,
        block_n: int,
        block_k: int,
        group_m: int,
        num_warps: int,
        num_stages: int,
    ) -> None:
        num_pid_n = triton.cdiv(n, block_n)
        grid = (triton.cdiv(m, block_m) * num_pid_n, )
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
            group_m=group_m if num_pid_n > 1 else 1,
            num_warps=num_warps,
            num_stages=num_stages,
        )

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

        self._launch(
            a_contig,
            b_contig,
            c,
            m,
            n,
            k,
            block_m=128,
            block_n=192,
            block_k=80,
            group_m=4,
            num_warps=8,
            num_stages=2,
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
