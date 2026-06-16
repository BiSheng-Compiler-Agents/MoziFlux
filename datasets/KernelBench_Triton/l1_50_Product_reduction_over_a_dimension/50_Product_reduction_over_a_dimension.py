import triton
import triton.language as tl


@triton.jit
def _prod_dim1_kernel(
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
    # Program IDs
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Offsets in the K dimension for this program
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    # Base pointer for this (b, k-block)
    base = pid_b * stride_b + offs_k * stride_k
    ptr = x_ptr + base

    # Provide compiler hints for better vectorization/coalescing
    tl.multiple_of(offs_k, 16)
    tl.max_contiguous(offs_k, BLOCK_K)

    # Use multiple independent accumulators to shorten dependency chains
    acc0 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    acc1 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    acc2 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)
    acc3 = tl.full([BLOCK_K], 1.0, dtype=tl.float32)

    m = 0
    # Main unrolled loop: process UNROLL rows per iteration when available
    while m + (UNROLL - 1) < M:
        # Load UNROLL rows; with mask on K only (rows guaranteed in-bounds here)
        v0 = tl.load(ptr + (m + 0) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        v1 = tl.load(ptr + (m + 1) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        v2 = tl.load(ptr + (m + 2) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        v3 = tl.load(ptr + (m + 3) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        v4 = tl.load(ptr + (m + 4) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        v5 = tl.load(ptr + (m + 5) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        v6 = tl.load(ptr + (m + 6) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)
        v7 = tl.load(ptr + (m + 7) * stride_m,
                     mask=mask_k,
                     other=1.0,
                     cache_modifier=".cg").to(tl.float32)

        # Pairwise products to improve ILP
        acc0 *= (v0 * v1)
        acc1 *= (v2 * v3)
        acc2 *= (v4 * v5)
        acc3 *= (v6 * v7)

        m += UNROLL

    # Tail handling
    while m < M:
        v = tl.load(ptr + m * stride_m,
                    mask=mask_k,
                    other=1.0,
                    cache_modifier=".cg").to(tl.float32)
        acc0 *= v
        m += 1

    # Combine accumulators
    out = (acc0 * acc1) * (acc2 * acc3)

    # Store the result
    out_ptrs = y_ptr + pid_b * stride_ob + offs_k * stride_ok
    tl.store(out_ptrs, out, mask=mask_k)
