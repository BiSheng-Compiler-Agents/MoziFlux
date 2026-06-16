import triton
import triton.language as tl


@triton.jit
def _fused_linear_sub_mul_relu_kernel(
    A_ptr,  # [M, K]
    W_ptr,  # [N, K] but accessed as [K, N] using strides
    B_ptr,  # [N]
    C_ptr,  # [M, N]
    SUB_VAL: tl.constexpr,  # scalar subtraction value
    MUL_VAL: tl.constexpr,  # scalar multiplication value
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,  # treat weight as [K, N] with these strides
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                          offs_k[None, :] * stride_ak)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk +
                          offs_n[None, :] * stride_wn)

        a_mask = (mask_m[:, None]) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (mask_n[None, :])

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(a, w)

    # Bias add
    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # Epilogue: (acc - SUB_VAL) * MUL_VAL then ReLU
    acc = (acc - SUB_VAL) * MUL_VAL
    acc = tl.maximum(acc, 0.0)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))
