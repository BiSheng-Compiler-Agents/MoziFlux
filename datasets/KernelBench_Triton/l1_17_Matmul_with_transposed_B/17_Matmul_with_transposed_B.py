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
                      num_stages=3),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_warps=8,
                      num_stages=3),
        triton.Config({
            "BLOCK_M": 32,
            "BLOCK_N": 256,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
        # Extra tensor-core friendly options for H200
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 256,
            "BLOCK_K": 128
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 128
        },
                      num_warps=4,
                      num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _a_bt_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
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
    # 2D program ids
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Alignment hints for better codegen
    tl.multiple_of(offs_k, 16)
    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)
    tl.static_assert(BLOCK_K % 16 == 0)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Build base pointers for first K tile
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk +
                      offs_n[None, :] * stride_bn)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K

        a = tl.load(a_ptrs,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                    cache_modifier=".cg")
        b = tl.load(b_ptrs,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                    cache_modifier=".cg")

        acc += tl.dot(a, b, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write back with masking
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])
