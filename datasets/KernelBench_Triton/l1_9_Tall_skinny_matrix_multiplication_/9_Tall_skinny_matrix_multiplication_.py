import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 16, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # 2D tile decomposition using a 1D grid with grouping along M dimension
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_block_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_block_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_mask = (k_iter + offs_k) < K
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)
        a = tl.load(A_block_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_block_ptrs, mask=b_mask, other=0.0)
        # Accumulate in fp32 for numerical stability; a/b may be fp16/bf16
        acc += tl.dot(a, b)
        k_iter += BLOCK_K
        A_block_ptrs += BLOCK_K * stride_ak
        B_block_ptrs += BLOCK_K * stride_bk

    C_block_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_block_ptrs, acc, mask=c_mask)
