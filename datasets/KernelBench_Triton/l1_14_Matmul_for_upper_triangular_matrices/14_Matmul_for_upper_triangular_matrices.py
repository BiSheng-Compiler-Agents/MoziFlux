import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline configs
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
        # Additional configs to better utilize H200
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 64
        },
                      num_warps=8,
                      num_stages=5),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_warps=8,
                      num_stages=5),
    ],
    key=["N"],
)
@triton.jit
def _upper_tri_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    stride_Am,
    stride_Ak,
    stride_Bk,
    stride_Bn,
    stride_Cm,
    stride_Cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D tile ids
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # Out-of-range tiles can be skipped early
    if (m0 >= N) or (n0 >= N):
        return
    # If the whole tile is strictly below the diagonal, we can skip it.
    if m0 > (n0 + BLOCK_N - 1):
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

    # K sweep using tiled dot-product.
    # Keep inputs in their native dtype to leverage tensor cores for fp16/bf16,
    # while accumulating in fp32. Disable TF32 to match PyTorch fp32 numerics.
    for k0 in range(0, N, BLOCK_K):
        k = k0 + rk
        k_in = k < N

        a_ptrs = A_ptr + (rm[:, None] * stride_Am + k[None, :] * stride_Ak)
        b_ptrs = B_ptr + (k[:, None] * stride_Bk + rn[None, :] * stride_Bn)

        a = tl.load(a_ptrs, mask=m_in[:, None] & k_in[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_in[:, None] & n_in[None, :], other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = C_ptr + (rm[:, None] * stride_Cm + rn[None, :] * stride_Cn)

    # Store only upper-triangular region
    tile_all_upper = (m0 + BLOCK_M - 1) <= n0
    full_in_bounds = (m0 + BLOCK_M) <= N and (n0 + BLOCK_N) <= N

    if tile_all_upper and full_in_bounds:
        # Fast path: no masks needed
        tl.store(c_ptrs, acc)
    else:
        store_mask = (rm[:, None] <= rn[None, :]) & m_in[:,
                                                         None] & n_in[None, :]
        tl.store(c_ptrs, acc, mask=store_mask)
