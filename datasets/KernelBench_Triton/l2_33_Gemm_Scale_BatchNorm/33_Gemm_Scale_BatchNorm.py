import triton
import triton.language as tl


@triton.jit
def _linear_scale_kernel(
    A_ptr,  # [M, K]
    B_ptr,  # stored as [N, K], accessed with strides to emulate [K, N]
    Bias_ptr,  # [N]
    Scale_ptr,  # [N]
    C_ptr,  # [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk +
                      offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_mask_a = (offs_m[:, None] < M) & (k_iter + offs_k[None, :] < K)
        k_mask_b = (k_iter + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b, other=0.0)
        acc += tl.dot(a, b)
        k_iter += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias and apply scale
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    scale = tl.load(Scale_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = acc + bias[None, :]
    acc = acc * scale[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
