import torch
import torch.nn as nn
import triton
import triton.language as tl


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _prod_dim1_generic_kernel(
    x_ptr,
    y_ptr,
    B,
    M,
    K,
    stride_b,
    stride_m,
    stride_k,
    stride_ob,
    stride_ok,
    BLOCK_K: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    ptr = x_ptr + pid_b * stride_b + offs_k * stride_k
    acc0 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    acc1 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    acc2 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    acc3 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    m = 0
    while m + 7 < M:
        r0 = tl.load(ptr + (m + 0) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        r1 = tl.load(ptr + (m + 1) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        r2 = tl.load(ptr + (m + 2) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        r3 = tl.load(ptr + (m + 3) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        r4 = tl.load(ptr + (m + 4) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        r5 = tl.load(ptr + (m + 5) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        r6 = tl.load(ptr + (m + 6) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        r7 = tl.load(ptr + (m + 7) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        acc0 *= r0 * r1
        acc1 *= r2 * r3
        acc2 *= r4 * r5
        acc3 *= r6 * r7
        m += UNROLL
    while m < M:
        acc0 *= tl.load(ptr + m * stride_m,
                        mask=mask_k,
                        other=1.0,
                        cache_modifier=".cg").to(tl.float32)
        m += 1
    out = (acc0 * acc1) * (acc2 * acc3)
    tl.store(y_ptr + pid_b * stride_ob + offs_k * stride_ok, out, mask=mask_k)


@triton.jit
def _prod_dim1_shape256_scalar_kernel(
    x_ptr,
    y_ptr,
    BLOCK_K: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    ptr = x_ptr + pid_b * 65536 + offs_k
    acc0 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    acc1 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    acc2 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    acc3 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    for m in tl.static_range(0, 256, UNROLL):
        r0 = tl.load(ptr + (m + 0) * 256, cache_modifier=".cg").to(tl.float32)
        r1 = tl.load(ptr + (m + 1) * 256, cache_modifier=".cg").to(tl.float32)
        r2 = tl.load(ptr + (m + 2) * 256, cache_modifier=".cg").to(tl.float32)
        r3 = tl.load(ptr + (m + 3) * 256, cache_modifier=".cg").to(tl.float32)
        acc0 *= r0
        acc1 *= r1
        acc2 *= r2
        acc3 *= r3
    out = (acc0 * acc1) * (acc2 * acc3)
    tl.store(y_ptr + pid_b * 256 + offs_k, out)


class ModelNew(nn.Module):

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return product_reduction_over_a_dimension(x, self.dim)


def product_reduction_over_a_dimension(x: torch.Tensor,
                                       dim: int) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "product_reduction_over_a_dimension expects an Ascend NPU tensor")
    if x.dim() != 3:
        raise ValueError(
            f"product_reduction_over_a_dimension expects a 3D tensor, got {x.dim()}D"
        )
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(f"unsupported dtype for product reduction: {x.dtype}")

    dim = dim if dim >= 0 else x.dim() + dim
    if dim != 1:
        raise NotImplementedError(
            f"product_reduction_over_a_dimension currently supports only dim=1/-2, got dim={dim}"
        )

    x_contig = x.contiguous()
    B, M, K = x_contig.shape
    y = torch.empty((B, K), device=x_contig.device, dtype=x_contig.dtype)
    if x_contig.is_contiguous() and M == 256 and K == 256:
        grid = (B, 2)
        _prod_dim1_shape256_scalar_kernel[grid](
            x_contig,
            y,
            BLOCK_K=128,
            UNROLL=4,
            num_warps=4,
            num_stages=4,
        )
        return y

    sB, sM, sK = x_contig.stride()
    oB, oK = y.stride()
    block_k = 128 if K >= 128 else 64
    grid = (B, triton.cdiv(K, block_k))
    _prod_dim1_generic_kernel[grid](
        x_contig,
        y,
        B,
        M,
        K,
        sB,
        sM,
        sK,
        oB,
        oK,
        BLOCK_K=block_k,
        UNROLL=8,
        num_warps=4 if block_k >= 128 else 2,
        num_stages=4,
    )
    return y


batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1


def get_inputs():
    device = "npu" if hasattr(torch,
                              "npu") and torch.npu.is_available() else "cpu"
    x = torch.randn(batch_size, dim1, dim2, device=device)
    return [x]


def get_init_inputs():
    return [reduction_dim]
