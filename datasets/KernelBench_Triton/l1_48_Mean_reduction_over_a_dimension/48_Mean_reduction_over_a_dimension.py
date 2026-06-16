import triton
import triton.language as tl


@triton.jit
def _mean_reduce_last_kernel(
    x_ptr,
    y_ptr,
    B,
    M,
    N,
    stride_b,
    stride_m,
    stride_n,
    y_stride_b,
    y_stride_m,
    invN,  # float32
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // M
    m = pid % M
    if (b >= B) | (m >= M):
        return

    base_ptr = x_ptr + b * stride_b + m * stride_m
    offs = tl.arange(0, BLOCK_N)

    # Accumulate across N in a vector register; reduce once at the end
    acc_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    k = 0
    UNROLL: tl.constexpr = 4
    while k < N:
        # Software unrolling for better ILP
        for u in tl.static_range(UNROLL):
            idx = k + u * BLOCK_N + offs
            mask = idx < N
            vals = tl.load(base_ptr + idx * stride_n, mask=mask, other=0.0)
            acc_vec += vals.to(tl.float32)
        k += UNROLL * BLOCK_N

    total = tl.sum(acc_vec, axis=0)
    mean = total * invN
    out_ptr = y_ptr + b * y_stride_b + m * y_stride_m
    tl.store(out_ptr, mean)


@triton.jit
def _mean_reduce_mid_tiled_kernel(
    x_ptr,
    y_ptr,
    B,
    M,
    N,
    stride_b,
    stride_m,
    stride_n,
    y_stride_b,
    y_stride_n,
    invM,  # float32
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)

    if b >= B:
        return

    n_start = n_block * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    m = 0
    UNROLL: tl.constexpr = 4
    # Unroll over M to reduce loop overhead; guard with mask for tail
    while m < M:
        for u in tl.static_range(UNROLL):
            mi = m + u
            mi_valid = mi < M
            ptr = x_ptr + b * stride_b + mi * stride_m + offs_n * stride_n
            vals = tl.load(ptr, mask=n_mask & mi_valid,
                           other=0.0).to(tl.float32)
            acc += vals
        m += UNROLL

    mean = acc * invM
    out_ptr = y_ptr + b * y_stride_b + offs_n * y_stride_n
    tl.store(out_ptr, mean, mask=n_mask)


@triton.jit
def _mean_reduce_first_tiled_kernel(
    x_ptr,
    y_ptr,
    B,
    M,
    N,
    stride_b,
    stride_m,
    stride_n,
    y_stride_m,
    y_stride_n,
    invB,  # float32
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)

    if m >= M:
        return

    n_start = n_block * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    b = 0
    UNROLL: tl.constexpr = 4
    # Unroll over B to improve ILP; guard tail with mask
    while b < B:
        for u in tl.static_range(UNROLL):
            bi = b + u
            bi_valid = bi < B
            ptr = x_ptr + bi * stride_b + m * stride_m + offs_n * stride_n
            vals = tl.load(ptr, mask=n_mask & bi_valid,
                           other=0.0).to(tl.float32)
            acc += vals
        b += UNROLL

    mean = acc * invB
    out_ptr = y_ptr + m * y_stride_m + offs_n * y_stride_n
    tl.store(out_ptr, mean, mask=n_mask)
