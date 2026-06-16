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
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_mul_leaky_kernel(
    A_ptr,  # [M, K]
    B_ptr,  # [N, K]  (weight)
    Bias_ptr,  # [N]
    C_ptr,  # [M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    multiplier,
    negative_slope,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_start * BLOCK_K
        k_mask = (k_offset + offs_k) < K
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                          (k_offset + offs_k)[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn +
                          (k_offset + offs_k)[:, None] * stride_bk)
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = (offs_n[None, :] < N) & k_mask[:, None]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    # Add bias [N] broadcast across M
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Multiply by scalar
    acc = acc * multiplier

    # LeakyReLU
    out = tl.where(acc >= 0, acc, acc * negative_slope)

    # Store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)
