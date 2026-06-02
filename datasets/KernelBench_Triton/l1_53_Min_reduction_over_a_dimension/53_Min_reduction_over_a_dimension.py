import triton
import triton.language as tl

@triton.jit
def _min_reduce_last_kernel(
    x_ptr, out_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    out_stride_b, out_stride_m,
    BLOCK_K: tl.constexpr,
):
    # Reduce over last dim (N). Each program computes one (b, m) output.
    pid = tl.program_id(axis=0)
    b = pid // M
    m = pid % M

    # Guard out-of-bounds programs (shouldn't happen if grid is correct, but keep safe)
    in_bounds = (b < B) & (m < M)
    if ~in_bounds:
        return

    base = b * stride_b + m * stride_m

    offs_k = tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    # Seed accumulator with first chunk and reduce to scalar
    k = 0
    idx0 = k + offs_k
    mask0 = idx0 < N
    ptrs0 = x_ptr + base + idx0 * stride_n
    x0 = tl.load(ptrs0, mask=mask0, other=float("inf"), cache_modifier=".ca")
    acc = tl.min(x0, axis=0)
    k += BLOCK_K

    # Unrolled loop to increase ILP and reduce loop overhead
    while k < N:
        for u in tl.static_range(0, 2):
            idx = k + offs_k + u * BLOCK_K
            mask = idx < N
            ptrs = x_ptr + base + idx * stride_n
            x = tl.load(ptrs, mask=mask, other=float("inf"), cache_modifier=".ca")
            acc = tl.minimum(acc, tl.min(x, axis=0))
        k += 2 * BLOCK_K

    out_off = b * out_stride_b + m * out_stride_m
    tl.store(out_ptr + out_off, acc)

@triton.jit
def _min_reduce_mid_kernel(
    x_ptr, out_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    out_stride_b, out_stride_n,
    BLOCK_K: tl.constexpr,
):
    # Reduce over middle dim (M). Each program computes one (b, n) output.
    pid = tl.program_id(axis=0)
    b = pid // N
    n = pid % N

    in_bounds = (b < B) & (n < N)
    if ~in_bounds:
        return

    base = b * stride_b + n * stride_n

    offs_k = tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    # Seed accumulator using first chunk -> scalar
    k = 0
    idx0 = k + offs_k
    mask0 = idx0 < M
    ptrs0 = x_ptr + base + idx0 * stride_m
    x0 = tl.load(ptrs0, mask=mask0, other=float("inf"), cache_modifier=".cg")
    acc = tl.min(x0, axis=0)
    k += BLOCK_K

    # Unrolled loop for strided loads
    while k < M:
        for u in tl.static_range(0, 2):
            idx = k + offs_k + u * BLOCK_K
            mask = idx < M
            ptrs = x_ptr + base + idx * stride_m
            x = tl.load(ptrs, mask=mask, other=float("inf"), cache_modifier=".cg")
            acc = tl.minimum(acc, tl.min(x, axis=0))
        k += 2 * BLOCK_K

    out_off = b * out_stride_b + n * out_stride_n
    tl.store(out_ptr + out_off, acc)

@triton.jit
def _min_reduce_first_kernel(
    x_ptr, out_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    out_stride_m, out_stride_n,
    BLOCK_K: tl.constexpr,
):
    # Reduce over first dim (B). Each program computes one (m, n) output.
    pid = tl.program_id(axis=0)
    m = pid // N
    n = pid % N

    in_bounds = (m < M) & (n < N)
    if ~in_bounds:
        return

    base = m * stride_m + n * stride_n

    offs_k = tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    # Seed accumulator using first chunk -> scalar
    k = 0
    idx0 = k + offs_k
    mask0 = idx0 < B
    ptrs0 = x_ptr + base + idx0 * stride_b
    x0 = tl.load(ptrs0, mask=mask0, other=float("inf"), cache_modifier=".cg")
    acc = tl.min(x0, axis=0)
    k += BLOCK_K

    # Unrolled loop for strided loads
    while k < B:
        for u in tl.static_range(0, 2):
            idx = k + offs_k + u * BLOCK_K
            mask = idx < B
            ptrs = x_ptr + base + idx * stride_b
            x = tl.load(ptrs, mask=mask, other=float("inf"), cache_modifier=".cg")
            acc = tl.minimum(acc, tl.min(x, axis=0))
        k += 2 * BLOCK_K

    out_off = m * out_stride_m + n * out_stride_n
    tl.store(out_ptr + out_off, acc)
