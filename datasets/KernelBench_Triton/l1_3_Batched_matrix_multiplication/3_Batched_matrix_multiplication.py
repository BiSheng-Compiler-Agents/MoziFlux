import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BATCH,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    # Compute block indices with correct grouping to improve L2 reuse
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for the current batch
    a_ptr_batch = a_ptr + bid * stride_ab
    b_ptr_batch = b_ptr + bid * stride_bb
    c_ptr_batch = c_ptr + bid * stride_cb

    # Accumulator in FP32 for numerical stability/consistency with torch
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_offs = k_iter + offs_k

        a_ptrs = a_ptr_batch + (offs_m[:, None] * stride_am +
                                k_offs[None, :] * stride_ak)
        b_ptrs = b_ptr_batch + (k_offs[:, None] * stride_bk +
                                offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        k_iter += BLOCK_K

    c_ptrs = c_ptr_batch + (offs_m[:, None] * stride_cm +
                            offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
