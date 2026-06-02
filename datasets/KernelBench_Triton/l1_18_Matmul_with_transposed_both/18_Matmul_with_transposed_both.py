import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_AT_BT_kernel(
    A_ptr,  # A: (K, M)
    B_ptr,  # B: (N, K)
    C_ptr,  # C: (M, N) = A.T @ B.T
    M,
    N,
    K,
    stride_a_k,
    stride_a_m,
    stride_b_n,
    stride_b_k,
    stride_c_m,
    stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)
    tl.multiple_of(offs_k, 16)
    tl.static_assert(BLOCK_K % 16 == 0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = A_ptr + (offs_k[:, None] * stride_a_k + offs_m[None, :] * stride_a_m)
    b_ptrs = B_ptr + (offs_n[None, :] * stride_b_n + offs_k[:, None] * stride_b_k)

    m_mask = offs_m < M
    n_mask = offs_n < N

    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a_km = tl.load(
            a_ptrs,
            mask=k_mask[:, None] & m_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        b_kn = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        acc += tl.dot(tl.trans(a_km), b_kn, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_a_k
        b_ptrs += BLOCK_K * stride_b_k

    c_ptrs = C_ptr + (offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n)
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])
