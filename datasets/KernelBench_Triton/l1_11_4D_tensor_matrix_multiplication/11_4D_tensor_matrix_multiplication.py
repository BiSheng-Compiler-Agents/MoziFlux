import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 256,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_stages=3,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=3,
                      num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_2d_kernel(
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
    # Program IDs for 2D tiling
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for rows/cols within the tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Pointer for the C tile
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)

    # Accumulator in fp32 (numerically stable, matches PyTorch semantics)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop along K dimension, blocked by BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # Build pointers for current A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                          k_range[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_range[:, None] * stride_bk +
                          offs_n[None, :] * stride_bn)

        # Masks for boundary checks
        a_mask = (offs_m[:, None] < M) & (k_range[None, :] < K)
        b_mask = (k_range[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles and upcast to fp32 for robust numerical agreement across dtypes
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Blocked matmul accumulate with strict FP32 math (disable TF32)
        acc += tl.dot(a, b, allow_tf32=False)

    # Write back results with boundary mask; pointer dtype of C determines cast
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
