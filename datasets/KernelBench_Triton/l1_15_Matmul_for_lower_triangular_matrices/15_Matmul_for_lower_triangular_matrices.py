import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_warps=4,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_warps=4,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_warps=4,
                      num_stages=4),
    ],
    key=["N"],
)
@triton.jit
def _lower_tri_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
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
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    if m0 >= N or n0 >= N:
        return
    if n0 > (m0 + BLOCK_M - 1):
        return

    rm = m0 + tl.arange(0, BLOCK_M)
    rn = n0 + tl.arange(0, BLOCK_N)
    m_in = rm < N
    n_in = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    rk = tl.arange(0, BLOCK_K)

    tl.multiple_of(rm, BLOCK_M)
    tl.multiple_of(rn, BLOCK_N)
    tl.multiple_of(rk, BLOCK_K)

    for k0 in range(0, N, BLOCK_K):
        k = k0 + rk
        k_in = k < N

        a_ptrs = A_ptr + (rm[:, None] * stride_am + k[None, :] * stride_ak)
        b_ptrs = B_ptr + (k[:, None] * stride_bk + rn[None, :] * stride_bn)

        a_mask = m_in[:, None] & k_in[None, :] & (k[None, :] <= rm[:, None])
        b_mask = k_in[:, None] & n_in[None, :] & (rn[None, :] <= k[:, None])

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tile_all_lower = (n0 + BLOCK_N - 1) <= m0
    full_in_bounds = (m0 + BLOCK_M) <= N and (n0 + BLOCK_N) <= N

    if tile_all_lower and full_in_bounds:
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty))
    else:
        store_mask = (rm[:, None] >= rn[None, :]) & m_in[:,
                                                         None] & n_in[None, :]
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=store_mask)
